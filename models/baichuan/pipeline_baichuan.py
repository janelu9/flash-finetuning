#!/usr/bin/env python
# coding: utf-8
# Created on Thur Jun 29 09:36:49 2023
# @author: Lu Jian
# Email:janelu@live.cn;

import torch
import transformers
from torch.nn import (
    Module,
    Embedding,
    Linear,
    CrossEntropyLoss,
)
from .modeling_baichuan import (
    BaichuanLayer,
    RMSNorm,
    _get_interleave   
)
from deepspeed.pipe import PipelineModule,LayerSpec
import json
import tqdm
import time
import os
import gc

class EmbeddingPipe(Embedding):
    
    def forward(self, inputs):
        inputs_embeds = super().forward(inputs)
        return (inputs_embeds,)

class DecoderLayerPipe(BaichuanLayer):

    def __init__(self, config, gradient_checkpointing=False):
        super().__init__(config)
        self.gradient_checkpointing = gradient_checkpointing
        self.n_head = config.num_attention_heads
        self.alibi_mask = None
        
    def _make_alibi_mask(self,hidden_states):
        if self.alibi_mask is None :
            _, seq_length, _ = hidden_states.shape
            dtype = hidden_states.dtype
            device = hidden_states.device
            slopes=torch.tensor(_get_interleave(self.n_head),device=device)
            alibi=(slopes.unsqueeze(1)* torch.arange(seq_length,device=device).unsqueeze(0)).unsqueeze(1)
            alibi_mask = torch.triu(torch.full([seq_length, seq_length],torch.finfo(dtype).min,device=device),1) + alibi
            self.alibi_mask = alibi_mask.to(dtype)
        return self.alibi_mask 
        
    def forward(self, inputs):      
        hidden_states,= inputs
        attention_mask = self._make_alibi_mask(hidden_states)
        if self.gradient_checkpointing and self.training:
            outputs = torch.utils.checkpoint.checkpoint(
                super().forward,
                hidden_states,
                attention_mask,
            )
        else:
            outputs = super().forward(
                hidden_states,
                attention_mask,
            )
        return (outputs[0], ) 

class RMSNormPipe(RMSNorm):
    
    def forward(self, inputs):
        hidden_states, = inputs
        return (super().forward(hidden_states),)

class LinearPipe(Linear):
    
    def forward(self, inputs):
        hidden_states, = inputs
        return super().forward(hidden_states)
    
class RMSNormHeadPipe(Module):
    
    def __init__(self,hidden_size,vocab_size,bias=True,eps=1e-6):
        super().__init__()
        self.norm = RMSNorm(hidden_size, epsilon=eps)
        self.lm_head = Linear(hidden_size, vocab_size, bias=bias)
        
    def forward(self, inputs):
        hidden_states, = inputs
        return self.lm_head(self.norm(hidden_states))

class CrossEntropyLoss(CrossEntropyLoss):
    
    def forward(self,logits, labels):
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        return super().forward(shift_logits, shift_labels)

class BaichuanForCausalLMPipe(PipelineModule):
    
    def __init__(self, config,gradient_checkpointing=False, **kwargs):
        self.config = config
        specs = [
            LayerSpec(EmbeddingPipe,
                      config.vocab_size,
                      config.hidden_size, 
                      config.pad_token_id),
            *[
                LayerSpec(DecoderLayerPipe,config,gradient_checkpointing) 
                for _ in range(config.num_hidden_layers)
            ],
            LayerSpec(RMSNormHeadPipe,
                      config.hidden_size,
                      config.vocab_size, 
                      bias=False,
                      eps=config.rms_norm_eps),
        ]
        super().__init__(layers=specs, **kwargs)
        
        pipe2hf = {
            '0.weight': "model.embed_tokens.weight",
        }
        
        layer_keys = [
            'model.layers.{i}.input_layernorm.weight',
            'model.layers.{i}.mlp.down_proj.weight',
            'model.layers.{i}.mlp.gate_proj.weight',
            'model.layers.{i}.mlp.up_proj.weight',
            'model.layers.{i}.post_attention_layernorm.weight',
            'model.layers.{i}.self_attn.W_pack.weight',
            'model.layers.{i}.self_attn.o_proj.weight'
        ]
        
        for i in range(self.config.num_hidden_layers):
            for k in layer_keys:
                hf_k = k.format(i=i)
                pipe_k = k.format(i=i+1)[13:]
                pipe2hf.update({pipe_k: hf_k})
            
        pipe2hf.update({
            f"{i+2}.norm.weight": "model.norm.weight",
            f"{i+2}.lm_head.weight": "lm_head.weight"
        })
        
        self.pipe2hf = {k:v for k,v in pipe2hf.items() if k in self.state_dict()}
        
        
    def from_pretrained(self,model_path):
        rank = torch.distributed.get_rank()
        stage_id = self.stage_id
        model_file = f"pytorch_model-{stage_id + 1:05d}-of-{self.num_stages:05d}.bin" if self.num_stages>1 else "pytorch_model.bin"
        stage_file = os.path.join(model_path,model_file)
        
        def check_and_replace_state_dict_keys(state_dict):
            assert set(self.pipe2hf.values()) == state_dict.keys(),"pretrained weight's keys not matched!"
            return {k:state_dict.pop(hf_k) for k,hf_k in self.pipe2hf.items()}
            
        try:
            self.load_state_dict(check_and_replace_state_dict_keys(torch.load(stage_file,map_location=torch.device('cpu'))))
        except :
            try:
                stage_dir = model_path+"-pipeline"
                stage_file = os.path.join(stage_dir,model_file)
                self.load_state_dict(check_and_replace_state_dict_keys(torch.load(stage_file,map_location=torch.device('cpu'))))
            except:
                if rank == 0:
                    print(f"cached model in {stage_dir}")
                    if not os.path.exists(stage_dir):
                        os.makedirs(stage_dir)
                    else:
                        os.system(f" rm -rf {stage_dir}/*")
                    self.config._name_or_path = stage_dir
                    self.config.transformers_version = transformers.__version__
                    self.config.to_json_file(os.path.join(stage_dir,"config.json"))
                    print(f"cached model in {stage_dir}")
                torch.distributed.barrier()
                #it seems mp doesn't work, model's parameters never been splited, 2D in fact -_-!
                if rank == self._topo.get_dim('model')*self._topo.get_dim('data')*stage_id:
                    state_dict={}
                    while not os.path.exists(stage_dir):
                        print(f"rank:{rank} is waiting for {stage_dir}")
                        time.sleep(1)
                    try:
                        model_index_json = os.path.join(model_path,
                                                        "pytorch_model.bin.index.json")
                        with open(model_index_json,"r") as f: 
                            weight_map = json.load(f)["weight_map"] 
                        for part in tqdm.tqdm({weight_map[hf_k] for _,hf_k in self.pipe2hf.items()}):
                            tmp = torch.load(os.path.join(model_path,part),
                                             map_location=torch.device('cpu'))
                            for _,hf_k in self.pipe2hf.items():
                                if hf_k in tmp:
                                    state_dict[hf_k]= tmp.pop(hf_k)
                            del tmp
                            gc.collect()
                        torch.save(state_dict,stage_file)                   
                    except:
                        tmp = torch.load(os.path.join(model_path,"pytorch_model.bin"),
                                         map_location=torch.device('cpu'))
                        for _,hf_k in self.pipe2hf.items():
                            state_dict[hf_k]= tmp.pop(hf_k)
                        torch.save(state_dict,stage_file)
                        del tmp
                    del state_dict
                    gc.collect()
                torch.distributed.barrier()
                self.load_state_dict(check_and_replace_state_dict_keys(torch.load(stage_file,map_location=torch.device('cpu'))))
            
    def save_pretrained(self,output_path):
        rank = torch.distributed.get_rank()
        stage_id = self.stage_id
        while not os.path.exists(output_path):
            print(f"rank:{rank} is waiting for {output_path}")
            time.sleep(1)
        if rank == 0:
            self.config._name_or_path = output_path
            self.config.transformers_version = transformers.__version__
            self.config.to_json_file(os.path.join(output_path,"config.json"))
        if rank == self._topo.get_dim('data')*stage_id:
            model_file = f"pytorch_model-{stage_id + 1:05d}-of-{self.num_stages:05d}.bin" if self.num_stages>1 else "pytorch_model.bin"
            print(f"saving the final model to {output_path}: {model_file}.")
            output_file = os.path.join(output_path,model_file)
            torch.save({self.pipe2hf[k]:v for k,v in self.state_dict().items() if k in self.pipe2hf},output_file)

