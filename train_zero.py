#!/usr/bin/env python
# coding: utf-8
# Created on Thur Jun 29 09:36:49 2023
# @author: Lu Jian
# Email:janelu@live.cn;

import torch
import deepspeed
from ds_utils import (
    print_rank_0,
    to_device,
    save_hf_format,
    set_random_seed,
    get_all_reduce_mean,
    get_optimizer_grouped_parameters,
    save_zero_three_model,
    load_hf_tokenizer,
    get_train_ds_config)
from data.utils import (
    shuffle_rank_0,
    PromptDataset,
    PromptDataCollator)
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,)
from model.lora import (
    convert_linear_layer_to_lora,
    convert_lora_to_linear_layer,
    only_optimize_lora_parameters)
# from transformers.deepspeed import HfDeepSpeedConfig
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import os
import gc
import pyarrow.parquet
import argparse

parser = argparse.ArgumentParser(description='My training script.')
parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')        
# Include DeepSpeed configuration arguments
parser.add_argument("--model",
                    type=str,
                    default= "openlm-research/open_llama_13b",
                    help="huggingface's model path")
parser.add_argument("--train-data",
                    type=str,
                    default= "",
                    help="data for training") 
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
args.zero_stage=3
args.num_train_epochs=1
args.per_device_train_batch_size = 1
args.gradient_accumulation_steps =1
args.seed=1234
args.weight_decay=0.01
args.lr_scheduler_type="cosine"
args.num_warmup_steps=0
args.learning_rate=1e-4
args.output_dir = "./output"
args.gradient_checkpointing = not args.only_optimize_lora

def main():
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        deepspeed.init_distributed()
    
    args.global_rank = torch.distributed.get_rank()
    args.world_size = torch.distributed.get_world_size()
    ds_config = get_train_ds_config(offload=args.offload,
                                    stage=args.zero_stage,
                                    max_out_tokens=args.max_len)
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * args.world_size * args.gradient_accumulation_steps
    ds_config['steps_per_print'] = args.steps_per_print
    set_random_seed(args.seed)
    torch.distributed.barrier()
    
    model = LlamaForCausalLM.from_pretrained(args.model)
    # dschf = HfDeepSpeedConfig(ds_config)
    # print_rank_0(dschf,args.global_rank)
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
                              
    if os.path.isfile(args.train_data):
        from convert_raws_to_ids import write_parquet
        cached_dir = os.path.splitext(os.path.basename(args.train_data))[0] + f"_{os.path.basename(args.model)}"
        write_parquet(args.train_data,cached_dir,args.model,MAX_SEQ_LENGTH=2048)
        args.train_data = cached_dir
    train_data_partitions = [os.path.join(args.train_data,f) for f in os.listdir(args.train_data) if os.path.isdir(os.path.join(args.train_data,f))]
    
    num_train_batch =sum(
        np.ceil(float(open(os.path.join(args.train_data,f)).read().split()[0])
                /args.per_device_train_batch_size
                /args.world_size)
        for f in os.listdir(args.train_data) if f[-4:] == '.crc')    
    num_update_steps_per_epoch = np.ceil(
        num_train_batch / args.gradient_accumulation_steps ) + len(train_data_partitions) - 1
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps = args.num_warmup_steps if args.num_warmup_steps >= 1 else int(args.num_warmup_steps * num_update_steps_per_epoch),
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,)       
    
    model,*_ = deepspeed.initialize(
        args=args,
        config=ds_config,
        model = model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)
        
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    def evaluation(model, eval_dataloader):
        model.eval()
        losses = 0
        for step, batch in enumerate(eval_dataloader):
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses += loss.float()
        losses = losses / (step + 1)
        try:
            perplexity = torch.exp(losses)
        except OverflowError:
            perplexity = float("inf")
        try:
            perplexity = get_all_reduce_mean(perplexity).item()
        except:
            pass
        return perplexity
    
    # Train!
    # print_rank_0("***** Running training *****", args.global_rank)
    # print_rank_0(
        # f"***** Evaluating perplexity, Epoch {0}/{args.num_train_epochs} *****",
        # args.global_rank)
    # perplexity = evaluation(model, eval_dataloader)
    # print_rank_0(f"ppl: {perplexity}", args.global_rank)
    
    for epoch in range(args.num_train_epochs):
        accumulation_train_batches = 0
        shuffle_rank_0(train_data_partitions,args.global_rank,epoch)
        for train_data_partition in train_data_partitions:
            try:
                train_data = pyarrow.parquet.read_table(train_data_partition)
                train_dataset = PromptDataset(
                    {k:train_data[k].to_numpy().tolist() 
                    for k in train_data.column_names})
            except:
                continue
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn = PromptDataCollator(),
                sampler = DistributedSampler(train_dataset),
                batch_size = args.per_device_train_batch_size)
            accumulation_train_batches += len(train_dataloader)
            print_rank_0(
                f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Data File: {train_data_partition}, Total Micro Batches: {accumulation_train_batches}/{int(num_train_batch)}",
                args.global_rank)
            print_rank_0(args, args.global_rank)
            model.train()
            for step, batch in enumerate(train_dataloader):
                batch = to_device(batch, device)
                outputs = model(**batch, use_cache=False)
                loss = outputs.loss
                model.backward(loss)
                model.step()
                if (step +1)%args.steps_per_print ==0:
                    print_rank_0(f"loss: {loss.item()}", args.global_rank)
            # Evaluate perplexity on the validation set.
            # perplexity = evaluation(model, eval_dataloader)
            del train_dataloader
            del train_dataset
            del train_data
            gc.collect()
            # Evaluate perplexity on the validation set.
            print_rank_0(
                f"***** Evaluating perplexity, Epoch {epoch+1}/{args.num_train_epochs} *****",
                args.global_rank)
            # perplexity = evaluation(model, eval_dataloader)
            # print_rank_0(f"ppl: {perplexity}", args.global_rank)
        model.tput_timer.update_epoch_count()
    
    if args.output_dir is not None:
        print_rank_0('saving the final model ...', args.global_rank)
        model = convert_lora_to_linear_layer(model)
        if args.global_rank == 0:
            save_hf_format(model, args)

        if args.zero_stage == 3:
            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            save_zero_three_model(model,
                                  args.global_rank,
                                  args.output_dir,
                                  zero_stage=args.zero_stage)
         
if __name__ == "__main__":
    main()
