#!/usr/bin/env python
# coding: utf-8
# Created on Thur Jun 29 09:36:49 2023
# @author: Lu Jian
# Email:janelu@live.cn;

import torch
import deepspeed
from transformers import (
    LlamaConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,)
from ds_utils import (
    print_rank_0,
    to_device,
    save_hf_format,
    set_random_seed,
    get_all_reduce_mean,
    get_optimizer_grouped_parameters,
    save_zero_three_model,
    load_hf_tokenizer,
    get_train_ds_config,
    PromptDataset,
    PromptDataCollatorPipe)
from lora import (
    convert_linear_layer_to_lora,
    convert_lora_to_linear_layer,
    only_optimize_lora_parameters)
from torch.utils.data import DataLoader
from deepspeed.utils import RepeatingLoader
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
from pipeline_llama import LlamaForCausalLMPipe,LlamaCrossEntropyLoss
import numpy as np
import pyarrow.parquet
import os
import argparse

parser = argparse.ArgumentParser(description='My training script.')
parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')        
# Include DeepSpeed configuration arguments
parser.add_argument('--offload',
                    action='store_true',
                    help='Enable ZeRO Offload techniques.')         
parser.add_argument('--max_len',
                    type=int,
                    default=2048,
                    help='max seq len')
parser.add_argument('--steps_per_print',
                    type=int,
                    default=10,
                    help='max seq len')              
parser.add_argument('--gradient_checkpointing',
                    action='store_true',
                    help='Enable gradient checkpointing for model.')
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
                    
parser = deepspeed.add_config_arguments(parser)

args=parser.parse_args()
args.model_path = "decapoda-research/llama-13b-hf"
args.data_dir = "news-commentary-v13-zh-en_parquet"
args.zero_stage=1
args.num_train_epochs=1
args.per_device_train_batch_size = 3
args.gradient_accumulation_steps = 2
args.seed=1234
args.weight_decay=0.01
args.lr_scheduler_type="cosine"
args.num_warmup_steps=500
args.learning_rate=1e-4
args.output_dir = "./output"
args.pipe_parallel_size = 1
args.model_parallel_size = 1
args.gradient_checkpointing = True

if args.gradient_checkpointing and args.lora_dim > 0:
    assert (
        not args.only_optimize_lora
    ), "--gradient_checkpointing and --only_optimize_lora cannot be enabled at the same time."

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

    ds_config = get_train_ds_config(
        offload=args.offload,
        stage=args.zero_stage,)
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size*args.gradient_accumulation_steps*args.data_parallel_size
    ds_config['steps_per_print'] = args.steps_per_print
    set_random_seed(args.seed)
    torch.distributed.barrier()

    config=LlamaConfig.from_pretrained(args.model_path)
    topo = PipeModelDataParallelTopology(
        num_pp = args.pipe_parallel_size,
        num_mp = args.model_parallel_size,
        num_dp = args.data_parallel_size)
    args.seed = args.seed + topo.get_coord(args.global_rank).pipe
    model = LlamaForCausalLMPipe(
        config,
        args.gradient_checkpointing,
        loss_fn=LlamaCrossEntropyLoss(),
        topology=topo,
        base_seed=args.seed,)                             
    model.from_pretrained(args.model_path)
    if args.lora_dim > 0:
        model = convert_linear_layer_to_lora(
            model,
            args.lora_module_name.split(","),
            args.lora_dim)
        if args.only_optimize_lora:
            model = only_optimize_lora_parameters(model)

    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay)
    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))
                              
    data_dir = args.data_dir
    data_files = [f for f in os.listdir(data_dir) if f[-4:] != '.crc']
    num_train_batch =sum(
        np.ceil(float(open(os.path.join(data_dir,f)).read().strip())/args.per_device_train_batch_size/args.data_parallel_size)
        for f in os.listdir(data_dir) if f[-4:] == '.crc') 
    num_update_steps_per_epoch = np.ceil(
        num_train_batch / args.gradient_accumulation_steps )
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,)      
    
    engine, *_ = deepspeed.initialize(
        args=args,
        config=ds_config,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,)

    for epoch in range(args.num_train_epochs):
        for data_file in data_files:
            data = pyarrow.parquet.read_table(os.path.join(data_dir,data_file))
            train_dataset = PromptDataset(
                {k:data[k].to_numpy().tolist() 
                 for k in data.column_names})
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=PromptDataCollatorPipe(),
                num_workers = args.gradient_accumulation_steps,
                shuffle=True,
                drop_last=False,
                batch_size=args.per_device_train_batch_size)
            cur_num_train_bacth =len(train_dataloader)
            print_rank_0(
                f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Data file: {data_file}, Total Micro Batches: {cur_num_train_bacth}/{int(num_train_batch)}",
                args.global_rank)
            print_rank_0(args, args.global_rank)
            train_loader = RepeatingLoader(train_dataloader)
            train_iter = iter(train_loader)
            for step in range(cur_num_train_bacth):
                loss = engine.train_batch(data_iter=train_iter)
            del data
            del train_dataset
            del train_dataloader
            gc.collect()
            
    if args.global_rank == 0:
        print_rank_0('saving model ...', args.global_rank)
        engine = convert_lora_to_linear_layer(engine)
        if not os.path.exist(args.output_dir):
            os.makedirs(args.output_dir)
        engine.save_fp16_model(args.output_dir)
         
if __name__ == "__main__":
    main()
