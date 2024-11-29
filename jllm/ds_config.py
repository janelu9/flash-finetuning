
GLOBAL_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 4


def get_train_ds_config(offload,
                        stage=2,
                        enable_hybrid_engine=False,
                        inference_tp_size=1,
                        release_inference_cache=False,
                        pin_parameters=True,
                        tp_gather_partition_size=8,
                        max_out_tokens=512):

    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        
        # "reduce_bucket_size": 500000000,
        # "reduce_scatter": True,
        # zero++ requires deepspeed>=0.10.0
        # "zero_quantized_weights": True,
        # "zero_hpz_partition_size": 16,
        # "zero_quantized_gradients": True,

        # "contiguous_gradients": True,
        # "overlap_comm": True,
        
        
        "offload_param": {
            "device": device,
        },
        "offload_optimizer": {
            "device": device,
        },
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 3e7,
        "stage3_prefetch_bucket_size": 3e7,
        "memory_efficient_linear": False
    }
    activation_checkpointing={
        "partition_activations": False,
        "cpu_checkpointing": False,
        "contiguous_memory_optimization": False,
        "number_checkpoints": None,
        "synchronize_checkpoint_boundary": False,
        "profile": False
    }
    return {
        "train_batch_size": GLOBAL_BATCH_SIZE,
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        "seed":1234,
        "steps_per_print": 1,
        "zero_allow_untested_optimizer": True,
        "zero_optimization": zero_opt_dict,
        "activation_checkpointing" :activation_checkpointing,
        # "fp16": {"enabled": True,"loss_scale_window": 100},
        "bf16": {"enabled": True},
        # "optimizer": {
            # "type": "AdamW",
            # "params": {
              # "lr": 1e-5,
              # "betas": [
                # 0.9,
                # 0.95
              # ],
              # "eps": 1e-8,
              # "weight_decay": 0.01
            # }
          # },
        # "scheduler": {
            # "type": "WarmupDecayLR",
            # "params": {
                # "warmup_min_lr": 1e-7,
                # "warmup_max_lr": 1e-5,
                # "warmup_num_steps": 300,
                # "total_num_steps":10000,
                # "warmup_type":'linear',
             # }
        # },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "hybrid_engine": {
            "enabled": enable_hybrid_engine,
            "max_out_tokens": max_out_tokens,
            "inference_tp_size": inference_tp_size,
            "release_inference_cache": release_inference_cache,
            "pin_parameters": pin_parameters,
            "tp_gather_partition_size": tp_gather_partition_size,
        }
    }
	
def get_eval_ds_config(offload, stage=0):
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": 1e4,
        "offload_param": {
            "device": device
        },
        "memory_efficient_linear": False
    }
    return {
        "train_batch_size": GLOBAL_BATCH_SIZE,
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "fp16": {
            "enabled": True
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False
    }
