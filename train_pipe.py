#!/usr/bin/env python
# coding: utf-8
# Created on Thur Jun 29 09:36:49 2023
# @author: Lu Jian
# Email:janelu@live.cn;

import torch
import deepspeed
from transformers import (
    AutoConfig,
    SchedulerType,
    get_scheduler,)
from easyllm.ds_utils import (
    set_random_seed,
    get_optimizer_grouped_parameters,
    get_train_ds_config)
from easyllm.model.lora import (
    convert_linear_layer_to_lora,
    only_optimize_lora_parameters) 
from easyllm.model import (
    ModelPipe,
    CrossEntropyLossPipe
    )
from easyllm.trainer import train
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.runtime.pipe import ProcessTopology
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description='My training script.')
parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')        
# Include DeepSpeed configuration arguments
parser.add_argument("--model",
                    type=str,
                    default= "baichuan-inc/Baichuan-13B-Chat",
                    help="huggingface's model path")
parser.add_argument("--train-data",
                    type=str,
                    default= "news-commentary-v13-zh-en_Baichuan-13B-Chat",
                    help="data for training")                       
parser.add_argument('--offload',
                    action='store_true',
                    help='Enable ZeRO Offload techniques.')         
parser.add_argument('--seq_length',
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
                    default=-1,
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
args.eval_data = ""
args.checkpoint_dir = "check"
args.from_ckpt = ""
args.resume_ckpt = ""
args.steps_per_checkpoint = -1
args.zero_stage=0
args.num_train_epochs=1
args.per_device_train_batch_size = 1
args.gradient_accumulation_steps = 2
args.seed=1234
args.weight_decay=0.01
args.lr_scheduler_type="cosine"
args.num_warmup_steps=100
args.learning_rate=3e-4
args.output_dir = "./output"
args.pipe_parallel_size = 1
args.model_parallel_size = 1
args.gradient_checkpointing = not args.only_optimize_lora
try:
    import flash_attn
    import xformers
    args.fast = True 
except:
    args.fast = False 

def main(args):
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
    if args.checkpoint_dir and not os.path.exists(args.checkpoint_dir) and args.global_rank ==0: 
        os.system(f"mkdir -p {args.checkpoint_dir}")
    if os.path.isfile(args.train_data) and args.global_rank ==0:
        from easyllm.data.convert_raw_to_ids import write_parquet
        cached_dir = os.path.splitext(os.path.basename(args.train_data))[0] + f"_{os.path.basename(args.model)}"
        write_parquet(args.train_data,cached_dir,args.model,MAX_SEQ_LENGTH=args.seq_length)
        args.train_data = cached_dir  
    train_data_partitions = [os.path.join(args.train_data,f) for f in os.listdir(args.train_data) if os.path.isdir(os.path.join(args.train_data,f))]
    if args.eval_data:
        if os.path.isfile(args.eval_data) and args.global_rank ==0:
            from easyllm.data.convert_raw_to_ids import write_parquet
            cached_dir = os.path.splitext(os.path.basename(args.eval_data))[0] + f"_{os.path.basename(args.model)}"
            write_parquet(args.eval_data,cached_dir,args.model,MAX_SEQ_LENGTH=args.seq_length)
            args.eval_data = cached_dir
        eval_data_partitions = [os.path.join(args.eval_data,f) for f in os.listdir(args.eval_data) if os.path.isdir(os.path.join(args.eval_data,f))]   
    torch.distributed.barrier()
    try:
        config = AutoConfig.from_pretrained(args.model,trust_remote_code=True)
    except:
        config = AutoConfig.from_pretrained(args.model)
    topo = ProcessTopology(['data','model','pipe'], [args.data_parallel_size, args.model_parallel_size, args.pipe_parallel_size])
    args.seed = args.seed + topo.get_coord(args.global_rank).pipe
    model = ModelPipe[config.model_type](
        config,
        args.gradient_checkpointing,
        args.fast,
        loss_fn=CrossEntropyLossPipe[config.model_type](),
        topology=topo,
        base_seed=args.seed,
        # partition_method="type:DecoderLayer",
        )
    if not(args.resume_ckpt or args.from_ckpt): model.from_pretrained(args.model)
    
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
                              
    '''
    How many folders, how many partitions. 
    If you want to load the data into memory at one time, moving all the parquet files to same folder. 
    That may cause "num_update_steps_per_epoch" to be un-precision. But it donesn't matter.
    ''' 
    num_train_batch =sum(
        np.ceil(float(open(os.path.join(args.train_data,f)).read().split()[0])/args.per_device_train_batch_size/args.data_parallel_size)
        for f in os.listdir(args.train_data) if f[-4:] == '.crc') 
    num_update_steps_per_epoch = np.ceil(
        num_train_batch / args.gradient_accumulation_steps ) + len(train_data_partitions) - 1
    args.num_training_steps = int(args.num_train_epochs * num_update_steps_per_epoch)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps if args.num_warmup_steps >= 1 else int(args.num_warmup_steps * num_update_steps_per_epoch),
        num_training_steps=args.num_training_steps)      
    
    engine, *_ = deepspeed.initialize(
        args=args,
        config=ds_config,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        )
      

    train(args,engine,train_data_partitions,eval_data_partitions if args.eval_data else None)
    

if __name__ == "__main__":
    main(args)
