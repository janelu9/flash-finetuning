#!/usr/bin/env python
# coding: utf-8
# Created on Thur Jun 29 09:36:49 2023
# @author: Lu Jian
# Email:janelu@live.cn;

import torch
import deepspeed
from transformers.deepspeed import HfDeepSpeedConfig
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
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
    PromptDataCollator
)
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
# from peft import (
    # prepare_model_for_int8_training,
    # LoraConfig,
    # get_peft_model,
    # get_peft_model_state_dict,
    # set_peft_model_state_dict,
# )
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
import gc
import pyarrow.parquet
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
                    
parser = deepspeed.add_config_arguments(parser)

args=parser.parse_args()
args.model_path = "decapoda-research/llama-micro-hf"
args.data_dir = "news-commentary-v13-zh-en_parquet"
args.zero_stage=3
args.num_train_epochs=1
args.per_device_train_batch_size = 2
args.gradient_accumulation_steps =1
args.seed=1234
args.weight_decay=0.01
args.lr_scheduler_type="cosine"
args.num_warmup_steps=500
args.learning_rate=1e-4
args.output_dir = "./output"
args.gradient_checkpointing = True

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
    
    model=LlamaForCausalLM.from_pretrained(args.model_path)
    dschf = HfDeepSpeedConfig(ds_config)
    print_rank_0(dschf,args.global_rank)
    

    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay)

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))
    data_dir = args.data_dir
    data_files = [f for f in os.listdir(data_dir) if f[-4:] != '.crc']
    num_train_batch =sum(
        np.ceil(float(open(os.path.join(data_dir,f)).read().strip())/args.per_device_train_batch_size/args.world_size)
        for f in os.listdir(data_dir) if f[-4:] == '.crc'
    ) 
    num_update_steps_per_epoch = np.ceil(
        num_train_batch / args.gradient_accumulation_steps 
        )
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )       
    
    model, _, _, _ = deepspeed.initialize(
    model = model,
    optimizer=optimizer,
    args=args,
    config=ds_config,
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
        for data_file in data_files:
            data = pyarrow.parquet.read_table(os.path.join(data_dir,data_file))
            train_dataset = PromptDataset({k:data[k].to_numpy().tolist() for k in data.column_names})
            train_dataloader = DataLoader(train_dataset,
                                          collate_fn = PromptDataCollator(),
                                          sampler = DistributedSampler(train_dataset),
                                          batch_size = args.per_device_train_batch_size)
            print_rank_0(
                f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Data File: {data_file}, Total Micro Batches: {len(train_dataloader)}/{int(num_train_batch)}",
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
            del data
            del train_dataset
            del train_dataloader
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
