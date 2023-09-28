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
    get_train_ds_config,)
from data.utils import (
    shuffle_rank_0,
    read_data)
from model.lora import (
    convert_linear_layer_to_lora,
    convert_lora_to_linear_layer,
    only_optimize_lora_parameters) 
from model import (
    ModelPipe,
    CrossEntropyLossPipe
    )
from torch.utils.data import DataLoader
from deepspeed.utils import RepeatingLoader
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.runtime.pipe import ProcessTopology
from tqdm import tqdm
import time
import numpy as np
import os
import gc
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
args.from_pretrained_checkpoint = ""
args.resume_dir = ""
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
    if args.checkpoint_dir and not os.path.exists(args.checkpoint_dir) and args.global_rank ==0: 
        os.system(f"mkdir -p {args.checkpoint_dir}")
    if os.path.isfile(args.train_data) and args.global_rank ==0:
        from convert_raw_to_ids import write_parquet
        cached_dir = os.path.splitext(os.path.basename(args.train_data))[0] + f"_{os.path.basename(args.model)}"
        write_parquet(args.train_data,cached_dir,args.model,MAX_SEQ_LENGTH=args.seq_length)
        args.train_data = cached_dir  
    train_data_partitions = [os.path.join(args.train_data,f) for f in os.listdir(args.train_data) if os.path.isdir(os.path.join(args.train_data,f))]
    if args.eval_data:
        if os.path.isfile(args.eval_data) and args.global_rank ==0:
            from convert_raw_to_ids import write_parquet
            cached_dir = os.path.splitext(os.path.basename(args.eval_data))[0] + f"_{os.path.basename(args.model)}"
            write_parquet(args.eval_data,cached_dir,args.model,MAX_SEQ_LENGTH=args.seq_length)
            args.eval_data = cached_dir
        eval_data_partitions = [os.path.join(args.eval_data,f) for f in os.listdir(args.eval_data) if os.path.isdir(os.path.join(args.eval_data,f))]   
    torch.distributed.barrier()
    try:
        config = AutoConfig.from_pretrained(args.model)
    except:
        config = AutoConfig.from_pretrained(args.model,trust_remote_code=True)
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
    if not(args.resume_dir or args.from_pretrained_checkpoint): model.from_pretrained(args.model)
    
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
    num_training_steps = int(args.num_train_epochs * num_update_steps_per_epoch)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps if args.num_warmup_steps >= 1 else int(args.num_warmup_steps * num_update_steps_per_epoch),
        num_training_steps=num_training_steps)      
    
    engine, *_ = deepspeed.initialize(
        args=args,
        config=ds_config,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        )
      
    checkpoint_memory = []
    skiped_epoch = 0
    skiped_partition_id = 0
    skiped_step = -1

    if args.resume_dir or args.from_pretrained_checkpoint:
        try:
            assert args.resume_dir
            _,ckpt_config=engine.load_checkpoint(args.resume_dir)
            assert ds_config['train_batch_size'] == ckpt_config["ds_config"]["train_batch_size"]
            skiped_epoch = ckpt_config["ds_config"]["epoch"]
            skiped_partition_id = ckpt_config["ds_config"]["partition_id"]
            skiped_step = ckpt_config["ds_config"]["step"]
            checkpoint_memory.append(engine.global_steps)
        except:
            try:
                if not args.from_pretrained_checkpoint:args.from_pretrained_checkpoint = args.resume_dir
                print_rank_0("Only model's weights are loaded.", args.global_rank)
                if engine.bfloat16_enabled():
                    engine._config.bfloat16_enabled = False
                    _,ckpt_config=engine.load_checkpoint(args.from_pretrained_checkpoint,load_module_only=True)
                    engine._config.bfloat16_enabled = True
                    engine.optimizer._restore_from_bit16_weights()
                else:
                    _,ckpt_config=engine.load_checkpoint(args.from_pretrained_checkpoint,load_module_only=True)
                skiped_epoch = ckpt_config["ds_config"].get("epoch",0)
                skiped_partition_id = ckpt_config["ds_config"].get("partition_id",-1) + 1         
            except:
                print_rank_0("No checkpoint's weights were loaded.",args.global_rank)
                
    accumulation_train_steps = engine.global_steps
    print_rank_0(args, args.global_rank)
    for epoch in range(args.num_train_epochs):
        if epoch < skiped_epoch:continue
        shuffle_rank_0(train_data_partitions,args.global_rank,epoch)
        for partition_id, train_data_partition in enumerate(train_data_partitions):
            if epoch == skiped_epoch and partition_id < skiped_partition_id:continue
            try:
                train_dataset,DataCollator,read_train_time = read_data(args,train_data_partition) 
            except:
                continue
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn = DataCollator(pad_token_id = config.pad_token_id),
                num_workers = min(int(os.cpu_count()*0.8),args.gradient_accumulation_steps//2 + 1),
                shuffle=True,
                drop_last=False,
                batch_size=args.per_device_train_batch_size)
            train_iter = iter(RepeatingLoader(train_dataloader))
            cur_num_train_bacth = int(np.ceil(len(train_dataloader)/args.data_parallel_size))
            start_step = 0 
            cur_train_bacth_steps = int(np.ceil(cur_num_train_bacth/args.gradient_accumulation_steps))
            if epoch == skiped_epoch and partition_id == skiped_partition_id :
                if skiped_step == cur_train_bacth_steps -1:
                    print_rank_0(f"Wash the memory of train data clean for {read_train_time} seconds ......",args.global_rank)
                    del train_iter; del train_dataloader; del train_dataset
                    gc.collect()
                    [time.sleep(read_train_time/100) for _ in tqdm(range(100))]
                    continue
                start_step = skiped_step + 1
                for _ in range(start_step):
                    next(train_iter)
            accumulation_train_steps += cur_train_bacth_steps - start_step
            print_rank_0(
                f"Beginning of Epoch: {epoch+1}/{args.num_train_epochs}\nPartition Rank: {partition_id+1}/{len(train_data_partitions)}\nPartition Name: {train_data_partition}\n"+
                f"Total Partition Steps: {accumulation_train_steps}/{num_training_steps}",
                args.global_rank)
            for step in range(start_step,cur_train_bacth_steps):
                loss = engine.train_batch(data_iter=train_iter)
                steps = engine.global_steps
                if args.eval_data and ((args.steps_per_eval>0 and steps % args.steps_per_checkpoint == 0) or steps == accumulation_train_steps):
                    engine.eval()
                    eval_loss = 0
                    num_samples = 0
                    for eval_data_partition in eval_data_partitions:
                        try:
                            eval_dataset,_,read_eval_time = read_data(args,eval_data_partition)
                        except:
                            continue
                        eval_dataloader = DataLoader(
                            eval_dataset,
                            collate_fn = DataCollator(pad_token_id = config.pad_token_id),
                            num_workers = min(int(os.cpu_count()*0.8),args.gradient_accumulation_steps//2 + 1),
                            shuffle=False,
                            drop_last=False,
                            batch_size=args.per_device_train_batch_size)
                        eval_iter = iter(RepeatingLoader(eval_dataloader))
                        cur_eval_bacth_steps = int(np.ceil(len(eval_dataloader)/args.data_parallel_size/args.gradient_accumulation_steps))
                        for eval_step in range(cur_eval_bacth_steps):
                            loss = engine.eval_batch(data_iter = eval_iter)
                            num_samples += 1
                            eval_loss += loss
                        print_rank_0(f"Wash the memory of eval data clean for {read_eval_time} seconds ......",args.global_rank)
                        engine.set_dataiterator(None)
                        del eval_iter;del eval_dataloader;del eval_dataset
                        gc.collect()
                        [time.sleep(read_eval_time/100) for _ in tqdm(range(100))]
                    print_rank_0(f"************************ eval loss: {eval_loss.item()/num_samples}************************ ",args.global_rank)
                    engine.train()
                if args.checkpoint_dir and ((args.steps_per_checkpoint>0 and steps % args.steps_per_checkpoint == 0) or steps == accumulation_train_steps):
                    if args.max_num_checkpoints > 0 and args.max_num_checkpoints == len(checkpoint_memory):
                        oldest = checkpoint_memory.pop(0)
                        os.system(f"rm -rf {os.path.join(args.checkpoint_dir,str(oldest))}")
                    engine.config["epoch"] = epoch
                    engine.config["partition_id"] = partition_id
                    engine.config["step"] = step
                    engine.save_checkpoint(args.checkpoint_dir,tag = steps)
                    checkpoint_memory.append(steps)
            print_rank_0(f"Wash the memory of train data clean for {read_train_time} seconds ......",args.global_rank)
            engine.set_dataiterator(None)
            del train_iter;del train_dataloader;del train_dataset
            gc.collect()
            [time.sleep(read_train_time/100) for _ in tqdm(range(100))]
            
    if args.checkpoint_dir and accumulation_train_steps != ([0]+checkpoint_memory)[-1]:
        if args.max_num_checkpoints>0 and args.max_num_checkpoints == len(checkpoint_memory):
            oldest = checkpoint_memory.pop(0)
            os.system(f"rm -rf {os.path.join(args.checkpoint_dir,str(oldest))}")
        engine.config["epoch"] = epoch
        engine.config["partition_id"] = partition_id
        engine.config["step"] = step
        engine.save_checkpoint(args.checkpoint_dir,tag = accumulation_train_steps)
        
    if args.output_dir:
        if not os.path.exists(args.output_dir) and args.global_rank == 0:
            os.makedirs(args.output_dir)
        convert_lora_to_linear_layer(engine.module).save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
