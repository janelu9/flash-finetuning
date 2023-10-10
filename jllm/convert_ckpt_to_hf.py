#!/usr/bin/env python
# coding: utf-8
# Created on Thur Jun 29 09:36:49 2023
# @author: Lu Jian
# Email:janelu@live.cn;

import torch
import deepspeed
from transformers import AutoConfig
from .model import (
    convert_linear_layer_to_lora,
    only_optimize_lora_parameters,
    convert_lora_to_linear_layer,
    ModelPipe,
    CrossEntropyLossPipe
    )
from deepspeed.runtime.pipe import ProcessTopology

import argparse

parser = argparse.ArgumentParser(description='model conversion script.')
parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')        
# Include DeepSpeed configuration arguments
parser.add_argument("--model",
                    type=str,
                    default= "baichuan-inc/Baichuan-13B-Chat",
                    help="huggingface's model path")
parser.add_argument('--pipe_parallel_size',
                    type=int,
                    default=1,
                    help='pipe parallel size')
parser.add_argument('--model_parallel_size',
                    type=int,
                    default=1,
                    help='model parallel size')
parser.add_argument('--no_bf16',
                    action='store_true',
                    help="parameters's dtype.")
parser.add_argument('--ckpt',
                    type=str,
                    help='checkpoint dir')
parser.add_argument('--tag',
                    type=str,
                    default=None,
                    help='checkpoint tag')
## LoRA for efficient training setting
parser.add_argument("--lora_dim",
                    type=int,
                    default=0,
                    help="If > 0, use LoRA for efficient training.")
parser.add_argument("--lora_module_name",
                    type=str,
                    default= "self_attn,mlp",
                    help="The scope of LoRA.")
parser.add_argument('--only_optimize_lora',
                    action='store_true',
                    help='Only optimize the LoRA parameters.')
parser.add_argument("--hf",
                    type=str,
                    default="output",
                    help="Where to store the model.")
parser = deepspeed.add_config_arguments(parser)

args=parser.parse_args()

def main():
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
    deepspeed.init_distributed()
    args.global_rank = torch.distributed.get_rank()
    args.world_size = torch.distributed.get_world_size()
    assert args.world_size % (args.pipe_parallel_size * args.model_parallel_size) == 0
    args.data_parallel_size = args.world_size // (args.pipe_parallel_size * args.model_parallel_size)
    topo = ProcessTopology(['data','model','pipe'], [args.data_parallel_size, args.model_parallel_size, args.pipe_parallel_size])
    try:
        config = AutoConfig.from_pretrained(args.model,trust_remote_code=True)
    except:
        config = AutoConfig.from_pretrained(args.model)
    with torch.no_grad():
        model = ModelPipe[config.model_type](
            config,
            loss_fn=CrossEntropyLossPipe[config.model_type](),
            topology=topo,
            # partition_method="type:DecoderLayer",
        )
        if args.lora_dim > 0:
            model = convert_linear_layer_to_lora(
                model,
                args.lora_module_name.split(","),
                args.lora_dim)
            if args.only_optimize_lora:
                model = only_optimize_lora_parameters(model)
        engine, *_ = deepspeed.initialize(
            args=args,
            config={"train_micro_batch_size_per_gpu":1,"bf16": {"enabled":not args.no_bf16}},
            model=model,
            )
        if engine.bfloat16_enabled():
            engine._config.bfloat16_enabled = False
            _,ckpt_config=engine.load_checkpoint(args.ckpt,tag=args.tag,load_module_only=True)
            engine._config.bfloat16_enabled = True
        else:
            _,ckpt_config=engine.load_checkpoint(args.ckpt,tag=args.tag,load_module_only=True)
        engine.eval()
        convert_lora_to_linear_layer(engine.module).save_pretrained(args.hf)

if __name__ == "__main__":
    main()
