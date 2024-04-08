#!/usr/bin/env python
# coding: utf-8
# Created on Thur Jun 29 09:36:49 2023
# @author: Jian Lu
# Email:janelu@live.cn;

import torch
import deepspeed
from transformers import (
    AutoConfig,
    SchedulerType,
    get_scheduler,)
from .utils import (
    set_random_seed,
    get_optimizer_grouped_parameters)
from .model import (
    convert_linear_layer_to_lora,
    only_optimize_lora_parameters,
    ModelPipe,
    )
from .trainer import train
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.runtime.pipe import ProcessTopology
import numpy as np
import os
import importlib
import datetime
import argparse
import sys
os.environ['PATH']+=":"+os.path.dirname(sys.executable)

parser = argparse.ArgumentParser(description='My training script.')
parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')        
# Include DeepSpeed configuration arguments
parser.add_argument("--model",
                    type=str,
                    default= "baichuan-inc/Baichuan-13B-Chat",
                    help="huggingface's model path")
parser.add_argument("--train_data",
                    type=str,
                    default= "",
                    help="data for training,jsonl or parqet folder")
parser.add_argument("--eval_data",
                    type=str,
                    default= "",
                    help="data for evalution,jsonl or parqet folder")
parser.add_argument("--from_ckpt",
                    type=str,
                    default= "",
                    help="ckpt dir to load pretrained model parameters")
parser.add_argument("--resume_ckpt",
                    type=str,
                    default= "",
                    help="ckpt dir to resume interruption")
parser.add_argument('--tag',
                    type=str,
                    default=None,
                    help='checkpoint tag')
parser.add_argument("--ds_config",
                    type=str,
                    default= "ds_config.py",
                    help="deepspeed's config file")
parser.add_argument('--zero_stage',
                    type=int,
                    default=0,
                    help='zero stage')
parser.add_argument('--timeout',
                    type=int,
                    default=1800,
                    help='timeout')
parser.add_argument('--pipe_parallel_size',
                    type=int,
                    default=1,
                    help='pipe parallel size')
parser.add_argument('--model_parallel_size',
                    type=int,
                    default=1,
                    help='model parallel size')
parser.add_argument('--offload',
                    action='store_true',
                    help='Enable ZeRO Offload techniques.') 
parser.add_argument('--num_train_epochs',
                    type=int,
                    default=1,
                    help='train epochs')
parser.add_argument('--per_device_train_batch_size',
                    type=int,
                    default=2,
                    help='per device train batch_size')
parser.add_argument('--gradient_accumulation_steps',
                    type=int,
                    default=1,
                    help='gradient accumulation steps')
parser.add_argument("--weight_decay",
                    type=float,
                    default=0.,
                    help="Weight decay to use.")
parser.add_argument("--lr_scheduler_type",
                    type=SchedulerType,
                    default="cosine",
                    help="The scheduler type to use.",
                    choices=[
                        "linear", "cosine", "cosine_with_restarts", "polynomial",
                        "constant", "constant_with_warmup"
                    ])
parser.add_argument("--num_warmup_steps",
                    type=float,
                    default=0,
                    help="Number or rate of steps for the warmup in the lr scheduler.")
parser.add_argument("--learning_rate",
                    type=float,
                    default=3e-4,
                    help= "Initial learning rate (after the potential warmup period) to use.",)
parser.add_argument('--seq_len',
                    type=int,
                    default=2048,
                    help='max seq len')
parser.add_argument('--block_mask',
                    action='store_true',
                    help="use BlockDiagonalCausalMask")
parser.add_argument('--steps_per_print',
                    type=int,
                    default=10,
                    help='steps per print')
parser.add_argument('--steps_per_eval',
                    type=int,
                    default=-1,
                    help='steps per eval')
parser.add_argument('--steps_per_checkpoint',
                    type=int,
                    default=-1,
                    help='steps per checkpoint')
parser.add_argument("--checkpoint",
                    type=str,
                    default= "",
                    help="checkpoint dir")
parser.add_argument('--background_executor',
                    type=str,
                    default='process',
                    choices=["process", "thread", "null", "none",""],
                    help='excutor of background')
parser.add_argument('--ckpt_step_gt',
                    type=int,
                    default=0,
                    help='checkpoint steps >= ckpt_step_gt')
parser.add_argument('--best_of',
                    type=int,
                    default=1,
                    help='checkpoint top k of eval_loss')
parser.add_argument('--ckpt_epoch',
                    type=str,
                    default="",
                    help='checkpoint the given epoches')
parser.add_argument('--skip_epoch',
                    type=str,
                    default="",
                    help='checkpoint except the given epoches')
parser.add_argument('--max_num_checkpoints',
                    type=int,
                    default=1,
                    help='max checkpoint num')
parser.add_argument('--only_ckpt_model',
                    action='store_true',
                    help='Only checkpoint the model parameters.')
parser.add_argument('--early_stop',
                    type=int,
                    default=-1,
                    help='if eval loss continuous rebound epoches == early_stop, training will be breaked')              
parser.add_argument('--no_gradient_checkpointing',
                    action='store_true',
                    help='disable gradient checkpointing for decoder layer.')
parser.add_argument('--low_mem',
                    action='store_true',
                    help='lower memory usage.')
parser.add_argument('--no_shuf',
                    action='store_true',
                    help='disable shuffle at every epoch.')
parser.add_argument('--no_safetensor',
                    action='store_true',
                    help='not use safetensor.')
parser.add_argument('--init',
                    action='store_true',
                    help='train from 0')
parser.add_argument("--seed",
                    type=int,
                    default=1234,
                    help="A seed for reproducible training.")
parser.add_argument("--output_dir",
                    type=str,
                    default="",
                    help="Where to store the model.")
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
get_train_ds_config = importlib.import_module(os.path.splitext(args.ds_config)[0]).get_train_ds_config
assert args.early_stop != 0
assert args.max_num_checkpoints != 0
assert args.best_of>0
if args.max_num_checkpoints<0:args.best_of=1

def main(args):
    args.local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    deepspeed.init_distributed(timeout=datetime.timedelta(seconds=args.timeout))
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
    if args.checkpoint and not os.path.exists(args.checkpoint) and args.global_rank ==0: 
        os.system(f"mkdir -p {args.checkpoint}")
    if os.path.isfile(args.train_data):
        cached_dir = os.path.join(os.path.dirname(args.train_data),os.path.splitext(os.path.basename(args.train_data))[0] + f"_{os.path.basename(args.model)}")
        if args.global_rank ==0:
            from .raw2ids import write_parquet
            write_parquet(args.train_data,cached_dir,args.model,MAX_SEQ_LENGTH=args.seq_len)
        torch.distributed.barrier()
        args.train_data = cached_dir
    train_data_partitions = [os.path.join(args.train_data,f) for f in os.listdir(args.train_data) if os.path.isdir(os.path.join(args.train_data,f))]
    if args.eval_data:
        if os.path.isfile(args.eval_data):
            cached_dir = os.path.join(os.path.dirname(args.eval_data),os.path.splitext(os.path.basename(args.eval_data))[0] + f"_{os.path.basename(args.model)}")
            if args.global_rank ==0: 
                from .raw2ids import write_parquet
                write_parquet(args.eval_data,cached_dir,args.model,MAX_SEQ_LENGTH=args.seq_len)
            torch.distributed.barrier()
            args.eval_data = cached_dir
        eval_data_partitions = [os.path.join(args.eval_data,f) for f in os.listdir(args.eval_data) if os.path.isdir(os.path.join(args.eval_data,f))]   
    
    try:
        config = AutoConfig.from_pretrained(args.model,trust_remote_code=True)
    except:
        config = AutoConfig.from_pretrained(args.model)
    config.block_mask=args.block_mask
    config.gradient_checkpointing=not args.no_gradient_checkpointing and not args.only_optimize_lora
    torch.distributed.barrier()
    
    topo = ProcessTopology(['data','pipe','model'], [args.data_parallel_size, args.pipe_parallel_size, args.model_parallel_size])
    args.seed = args.seed + topo.get_coord(args.global_rank).pipe
    
    if args.model_parallel_size > 1:
        from jllm.core import parallel_state,tensor_parallel
        parallel_state.initialize_model_parallel(args.model_parallel_size,args.pipe_parallel_size)
        tensor_parallel.model_parallel_cuda_manual_seed(args.seed)
        from jllm.core.model_parallel_config import ModelParallelConfig
        parallel_config = ModelParallelConfig(tensor_model_parallel_size=args.model_parallel_size,
                                              pipeline_model_parallel_size=args.pipe_parallel_size,
                                              params_dtype=config.torch_dtype,
                                              pipeline_dtype=config.torch_dtype
                                             )
        parallel_config.batch_size = args.per_device_train_batch_size
        parallel_config.seq_length = int(open(os.path.join(args.train_data,[f for f in os.listdir(args.train_data) if f[-4:] == '.crc'][0])).read().split()[1])
        parallel_config.low_mem = args.low_mem
        from jllm.model import ModelParallel
        model =  ModelParallel[config.architectures[0]](
            config,
            parallel_config=parallel_config,
            topology=topo,
            base_seed=args.seed,
            # partition_method="type:DecoderLayer",
            )
    else:
        model = ModelPipe[config.architectures[0]](
            config,
            topology=topo,
            base_seed=args.seed,
            # partition_method="type:DecoderLayer",
            )
        
    if not(args.resume_ckpt or args.from_ckpt) and not args.init: model.from_pretrained(args.model)
    
    if args.lora_dim > 0:
        model = convert_linear_layer_to_lora(
            model,
            args.lora_module_name.split(","),
            args.lora_dim)
        if args.only_optimize_lora:
            model = only_optimize_lora_parameters(model)
    if "optimizer" not in ds_config:
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
    if 'scheduler' not in ds_config:
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps if args.num_warmup_steps >= 1 else int(args.num_warmup_steps * num_update_steps_per_epoch),
            num_training_steps=args.num_training_steps)      
    
    engine, *_ = deepspeed.initialize(
        args=args,
        config=ds_config,
        model=model,
        optimizer=optimizer if "optimizer" not in ds_config else None,
        lr_scheduler=lr_scheduler if "scheduler" not in ds_config else None,
        )
      

    train(args,engine,train_data_partitions,eval_data_partitions if args.eval_data else None)
    

if __name__ == "__main__":
    main(args)
