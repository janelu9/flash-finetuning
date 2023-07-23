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
import gc
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
                    help='steps per print')
parser.add_argument('--steps_per_eval',
                    type=int,
                    default=100,
                    help='steps per eval')
parser.add_argument('--steps_per_checkpoint',
                    type=int,
                    default=100,
                    help='steps per checkpoint')
parser.add_argument("--checkpoint_dir",
                    type=str,
                    default= "",
                    help="checkpoint dir")
parser.add_argument('--max_num_checkpoints',
                    type=int,
                    default=-1,
                    help='max checkpoint num')
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
args.model_path = "openlm-research/open_llama_13b"
args.train_data_dir = "news-commentary-v13-zh-en_parquet"
args.eval_data_dir = ""
args.zero_stage=1
args.num_train_epochs=1
args.per_device_train_batch_size = 2
args.gradient_accumulation_steps = 2
args.seed=1234
args.weight_decay=0.01
args.lr_scheduler_type="cosine"
args.num_warmup_steps=50
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
    if args.checkpoint_dir and not os.path.exists(args.checkpoint_dir) and args.global_rank ==0 :
        os.system(f"mkdir -p {args.checkpoint_dir}")
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
                              
    train_data_dir = args.train_data_dir
    data_files = [f for f in os.listdir(train_data_dir) if f[-4:] != '.crc']
    num_train_batch =sum(
        np.ceil(float(open(os.path.join(train_data_dir,f)).read().strip())/args.per_device_train_batch_size/args.data_parallel_size)
        for f in os.listdir(train_data_dir) if f[-4:] == '.crc') 
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
    
    if args.eval_data_dir:
        eval_data_files = [f for f in os.listdir(args.eval_data_dir) if f[-4:] != '.crc']
        
    checkpoint_memory=[]
    for epoch in range(args.num_train_epochs):
        accumulation_train_batches = 0
        for data_file in data_files:
            data = pyarrow.parquet.read_table(os.path.join(train_data_dir,data_file))
            train_dataset = PromptDataset(
                {k:data[k].to_numpy().tolist() 
                 for k in data.column_names})
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=PromptDataCollatorPipe(),
                num_workers = min(int(os.cpu_count()*0.8),args.gradient_accumulation_steps//2 + 1),
                shuffle=True,
                drop_last=False,
                batch_size=args.per_device_train_batch_size)
            cur_num_train_bacth =int(np.ceil(len(train_dataloader)/args.data_parallel_size))
            accumulation_train_batches += cur_num_train_bacth
            print_rank_0(
                f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Data file: {data_file}, Total Micro Batches: {accumulation_train_batches}/{int(num_train_batch)}",
                args.global_rank)
            print_rank_0(args, args.global_rank)
            train_loader = RepeatingLoader(train_dataloader)
            train_iter = iter(train_loader)
            cur_train_bacth_steps = int(np.ceil(cur_num_train_bacth/args.gradient_accumulation_steps))+20#few steps may be skipped
            for step in range(cur_train_bacth_steps):
                loss = engine.train_batch(data_iter=train_iter)
                steps = engine.global_steps
                if args.eval_data_dir and steps % args.steps_per_eval == 0:
                    engine.eval()
                    eval_loss = 0
                    num_samples = 0
                    for eval_data_file in eval_data_files:
                        eval_data = pyarrow.parquet.read_table(os.path.join(args.eval_data_dir,eval_data_file))
                        eval_dataset = PromptDataset(
                            {k:eval_data[k].to_numpy().tolist()
                             for k in eval_data.column_names})
                        eval_dataloader = DataLoader(
                            eval_dataset,
                            collate_fn=PromptDataCollatorPipe(),
                            num_workers = min(int(os.cpu_count()*0.8),args.gradient_accumulation_steps//2 + 1),
                            shuffle=False,
                            drop_last=False,
                            batch_size=args.per_device_train_batch_size)
                        eval_loader = RepeatingLoader(eval_dataloader)
                        eval_iter = iter(eval_loader)
                        cur_eval_bacth_steps = int(np.ceil(len(eval_dataloader)/args.data_parallel_size/args.gradient_accumulation_steps))
                        for eval_step in range(cur_eval_bacth_steps):
                            loss = engine.eval_batch(data_iter = eval_iter)
                            num_samples += 1
                            eval_loss += loss
                        del eval_data
                        del eval_dataset
                        del eval_dataloader
                        gc.collect()
                    print_rank_0(f"************************ eval loss: {eval_loss/num_samples}************************ ",args.global_rank)
                    engine.train()
                if args.checkpoint_dir and steps % args.steps_per_checkpoint == 0:
                    if args.max_num_checkpoints > 0 and args.max_num_checkpoints == len(checkpoint_memory):
                        oldest = checkpoint_memory.pop(0)
                        os.system(f"rm -rf {os.path.join(args.checkpoint_dir,str(oldest))}")
                    engine.save_checkpoint(args.checkpoint_dir,tag=steps)
                    checkpoint_memory.append(steps)
            del data
            del train_dataset
            del train_dataloader
            gc.collect()
            
    if args.checkpoint_dir:
        if args.max_num_checkpoints>0 and args.max_num_checkpoints == len(checkpoint_memory):
            oldest = checkpoint_memory.pop(0)
            os.system(f"rm -rf {os.path.join(args.checkpoint_dir,str(steps))}")
        engine.save_checkpoint(args.checkpoint_dir,tag=steps)
        
    if args.output_dir:
        if not os.path.exists(args.output_dir) and args.global_rank == 0:
            os.makedirs(args.output_dir)
        convert_lora_to_linear_layer(engine.module).half().save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
