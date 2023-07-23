#!/usr/bin/env python
# coding: utf-8
# Created on Sat Jul 14 21:22:49 2023
# @author: Lu Jian
# Email:janelu@live.cn;

import torch.nn.functional as F
import torch

def beam_search(self,input_ids,max_length =1024,num_beams =15):
    # Adopted from https://gitee.com/janelu9/TrOCR/blob/main/image2text.py#L838 
    # 1.6x faster than huggingface
    INF = 1.e4
    
    def prepare(input_ids):
        out,cache= self.model(input_ids,use_cache = True,return_dict =False)
        logits = self.lm_head(out[:,-1,:])
        log_probs = F.log_softmax(logits,dim=-1)
        cur_log_probs,cur_word = log_probs.topk(num_beams)
        cur_seqs = cur_word.clone().unsqueeze(-1)
        end = cur_word == self.config.eos_token_id
        end_flags = end.float()
        end_log_probs = torch.full_like(cur_log_probs,-INF)
        end_word = torch.full_like(cur_seqs,self.config.eos_token_id)
        end_seqs = end_word.clone()
        end_log_probs_pre = end_log_probs.clone()
        end_flags_pre = torch.zeros_like(end_flags)
        if torch.any(end):
            end_log_probs = cur_log_probs+(1.-end_flags)*-INF
            cur_log_probs += end_flags*-INF
        device = logits.device
        batch_size,pro_len = input_ids.size()
        length = torch.tensor([1],device=device)
        position_id = torch.tensor([[pro_len]],device=device)
        dim = cache[0][0].shape[1:]
        cache = [(k.unsqueeze(1).tile(1,num_beams,1,1,1).view(-1,*dim),v.unsqueeze(1).tile(1,num_beams,1,1,1).view(-1,*dim)) for k,v in cache]
        vocab_size = self.model.vocab_size
        batch_pos = (torch.arange(batch_size,device=device)).unsqueeze(1)
#         batch_pos_ = batch_pos.tile(1,num_beams)
        batch_pos_ = batch_pos.tile(1,num_beams+1)
        batch_pos =  batch_pos.tile(1,num_beams)
        batch_beam_pos = batch_pos * num_beams
        return cur_log_probs,cur_seqs,cur_word,cache,end_log_probs,end_seqs,end_word,end_flags,end_log_probs_pre,end_flags_pre,vocab_size,batch_size,batch_pos,batch_pos_,batch_beam_pos,length,position_id
    
    def stop(cur_log_probs,end_log_probs_pre,end_flags_pre):
        max_cur_log_probs = cur_log_probs.max(1)[0]
        min_end_log_probs = (end_log_probs_pre*end_flags_pre).min(1)[0]
        min_end_log_probs += (1. - end_flags_pre.max(1)[0]) * -INF   
        return torch.all(min_end_log_probs>max_cur_log_probs)
    
    def step(cur_word,position_id,cache,cur_log_probs,end_seqs):        
        out,cache= self.model(cur_word.contiguous().view(-1,1),position_ids = position_id,past_key_values = cache,use_cache = True,return_dict =False)
        logits = self.lm_head(out[:,-1,:])
        log_probs = F.log_softmax(logits,dim=-1)
        cur_log_probs =(log_probs +cur_log_probs.contiguous().view(-1,1)*(length.log()+1))/((length+1).log()+1)
        cur_log_probs,topk_ids = cur_log_probs.view(batch_size,-1).topk(num_beams+1)
#         cur_log_probs = log_probs +cur_log_probs.contiguous().view(-1,1)
#         cur_log_probs,topk_ids = cur_log_probs.view(batch_size,-1).topk(num_beams)
        cur_word =   (topk_ids % vocab_size)
        beam_coordinate = topk_ids // vocab_size
        end_seqs = torch.cat([end_seqs,end_word],-1)
        end = cur_word == self.config.eos_token_id
        return cache,cur_log_probs,cur_word,beam_coordinate,end_seqs,end
    
    def update(end_log_probs,end_flags,end,end_seqs,cur_seqs,beam_coordinate,cur_word,cur_log_probs,cache):
        end_log_probs_pre = end_log_probs
        end_flags_pre = end_flags
        cur_end_flags = end.float()
        cur_end_seqs = torch.cat([cur_seqs[batch_pos_,beam_coordinate],cur_word.unsqueeze(-1)],-1)
        combine_seqs =  torch.cat([end_seqs, cur_end_seqs], 1)
        combine_flags = torch.cat([end_flags_pre, cur_end_flags], 1)
        combine_log_probs =torch.cat([end_log_probs, cur_log_probs+(1.-cur_end_flags)*-INF],1)
        end_log_probs, topk_ids = combine_log_probs.topk(num_beams)
        end_seqs = combine_seqs[batch_pos,topk_ids]
        end_flags = combine_flags[batch_pos,topk_ids]
        cur_log_probs += cur_end_flags *-INF
        cur_log_probs,topk_ids = cur_log_probs.topk(num_beams)
        cur_word = cur_word.gather(1,topk_ids)
        cur_seqs = torch.cat([cur_seqs[batch_pos,topk_ids],cur_word.unsqueeze(-1)],-1)
        select_index = (batch_beam_pos + beam_coordinate[batch_pos,topk_ids]).view(-1)
        cache = [(k[select_index],v[select_index]) for k,v in cache]
        return end_log_probs_pre,end_flags_pre,end_log_probs,end_seqs,end_flags,cur_log_probs,cur_word,cur_seqs,cache
    
    def grow(cur_log_probs,cur_word,cur_seqs,beam_coordinate,cache):
        cur_log_probs = cur_log_probs[:,:num_beams]
        cur_word = cur_word[:,:num_beams]
        cur_seqs = torch.cat([cur_seqs[batch_pos,beam_coordinate[:,:num_beams]],cur_word.unsqueeze(-1)],-1)
        select_index = (batch_beam_pos + beam_coordinate[:,:num_beams]).view(-1)
        cache = [(k[select_index],v[select_index]) for k,v in cache]
        return cur_log_probs,cur_word,cur_seqs,cache
    
    def final(end_seqs,end_log_probs,cur_seqs,cur_log_probs):
        combine_seqs =  torch.cat([end_seqs, cur_seqs], 1)
        combine_log_probs =torch.cat([end_log_probs, cur_log_probs],1)
        final_log_probs, topk_ids = combine_log_probs.topk(num_beams)
        final_seqs = combine_seqs[batch_pos,topk_ids]
        return final_seqs,final_log_probs
    
    with torch.no_grad():
        cur_log_probs,cur_seqs,cur_word,cache,end_log_probs,end_seqs,end_word,end_flags,end_log_probs_pre,end_flags_pre,vocab_size,batch_size,batch_pos,batch_pos_,batch_beam_pos,length,position_id = prepare(input_ids)
        while position_id < max_length - 1 and not stop(cur_log_probs,end_log_probs_pre,end_flags_pre):
            cache,cur_log_probs,cur_word,beam_coordinate,end_seqs,end = step(cur_word,position_id,cache,cur_log_probs,end_seqs)
            if torch.any(end):
                end_log_probs_pre,end_flags_pre,end_log_probs,end_seqs,end_flags,cur_log_probs,cur_word,cur_seqs,cache = update(end_log_probs,end_flags,end,end_seqs,cur_seqs,beam_coordinate,cur_word,cur_log_probs,cache)
            else:
                cur_log_probs,cur_word,cur_seqs,cache = grow(cur_log_probs,cur_word,cur_seqs,beam_coordinate,cache)
            position_id += 1
            length += 1
        return final(end_seqs,end_log_probs,cur_seqs,cur_log_probs)
        

def sample(self, input_ids, max_length = 1024, num_beams = 1, top_k = 50, top_p = 1, T = 1):

    INF = 1.e4
    
    def prepare(input_ids):
        out,cache= self.model(input_ids,use_cache = True,return_dict =False)
        logits = self.lm_head(out[:,-1,:])
        log_probs = F.log_softmax(logits,dim=-1)
        cur_log_probs,cur_word = log_probs.topk(top_k)
        cur_topk_probs = cur_log_probs.softmax(-1)
        device = logits.device
        batch_size,pro_len = input_ids.size()
        left_pad = torch.full((batch_size,1),False,dtype=torch.bool,device=device)
        cur_topk_probs = (cur_topk_probs.masked_fill_(torch.cat([left_pad,cur_topk_probs.cumsum(-1)[:,:-1] > top_p],-1),-INF)/T).softmax(-1)
        topb_ids = torch.multinomial(cur_topk_probs,num_beams)
        cur_log_probs = cur_log_probs.gather(1,topb_ids)
        cur_word = cur_word.gather(1,topb_ids)
        cur_seqs = cur_word.clone().unsqueeze(-1)
        end = cur_word == self.config.eos_token_id
        end_flags = end.float()
        end_log_probs = torch.full_like(cur_log_probs,-INF)
        end_word = torch.full_like(cur_seqs,self.config.eos_token_id)
        end_seqs = end_word.clone()
        end_log_probs_pre = end_log_probs.clone()
        end_flags_pre = torch.zeros_like(end_flags)
        if torch.any(end):
            end_log_probs = cur_log_probs+(1.-end_flags)*-INF
            cur_log_probs += end_flags*-INF
        length = torch.tensor([1],device=device)
        position_id = torch.tensor([[pro_len]],device=device)
        dim = cache[0][0].shape[1:]
        cache = [(k.unsqueeze(1).tile(1,num_beams,1,1,1).view(-1,*dim),v.unsqueeze(1).tile(1,num_beams,1,1,1).view(-1,*dim)) for k,v in cache]
        vocab_size = self.model.vocab_size
        batch_pos = (torch.arange(batch_size,device=device)).unsqueeze(1)
#         batch_pos_ = batch_pos.tile(1,num_beams)
        batch_pos_ = batch_pos.tile(1,num_beams+1)
        batch_pos =  batch_pos.tile(1,num_beams)
        batch_beam_pos = batch_pos * num_beams
        return cur_log_probs,cur_seqs,cur_word,cache,end_log_probs,end_seqs,end_word,end_flags,end_log_probs_pre,end_flags_pre,vocab_size,batch_size,batch_pos,batch_pos_,batch_beam_pos,length,position_id,left_pad
    
    def stop(cur_log_probs,end_log_probs_pre,end_flags_pre):
        max_cur_log_probs = cur_log_probs.max(1)[0]
        min_end_log_probs = (end_log_probs_pre*end_flags_pre).min(1)[0]
        min_end_log_probs += (1. - end_flags_pre.max(1)[0]) * -INF   
        return torch.all(min_end_log_probs>max_cur_log_probs)
    
    def step(cur_word,position_id,cache,cur_log_probs,end_seqs):        
        out,cache= self.model(cur_word.contiguous().view(-1,1),position_ids = position_id,past_key_values = cache,use_cache = True,return_dict =False)
        logits = self.lm_head(out[:,-1,:])
        log_probs = F.log_softmax(logits,dim=-1)
        cur_log_probs =(log_probs +cur_log_probs.contiguous().view(-1,1)*(length.log()+1))/((length+1).log()+1)
        cur_log_probs, topk_ids = cur_log_probs.view(batch_size,-1).topk(top_k)
        cur_word =   (topk_ids % vocab_size)
        beam_coordinate = topk_ids // vocab_size
        cur_topk_probs = cur_log_probs.softmax(-1)
        cur_topk_probs = (cur_topk_probs.masked_fill_(torch.cat([left_pad,cur_topk_probs.cumsum(-1)[:,:-1] > top_p],-1),-INF)/T).softmax(-1)
        topb_ids = torch.multinomial(cur_topk_probs,num_beams + 1 )
        cur_log_probs = cur_log_probs.gather(1,topb_ids)
        cur_word = cur_word.gather(1,topb_ids)
        beam_coordinate = beam_coordinate.gather(1,topb_ids)
        end_seqs = torch.cat([end_seqs,end_word],-1)
        end = cur_word == self.config.eos_token_id
        return cache,cur_log_probs,cur_word,beam_coordinate,end_seqs,end
    
    def update(end_log_probs,end_flags,end,end_seqs,cur_seqs,beam_coordinate,cur_word,cur_log_probs,cache):
        end_log_probs_pre = end_log_probs
        end_flags_pre = end_flags
        cur_end_flags = end.float()
        cur_end_seqs = torch.cat([cur_seqs[batch_pos_,beam_coordinate],cur_word.unsqueeze(-1)],-1)
        combine_seqs =  torch.cat([end_seqs, cur_end_seqs], 1)
        combine_flags = torch.cat([end_flags_pre, cur_end_flags], 1)
        combine_log_probs =torch.cat([end_log_probs, cur_log_probs+(1.-cur_end_flags)*-INF],1)
        end_log_probs, topk_ids = combine_log_probs.topk(num_beams)
        end_seqs = combine_seqs[batch_pos,topk_ids]
        end_flags = combine_flags[batch_pos,topk_ids]
        cur_log_probs += cur_end_flags *-INF
        cur_log_probs,topk_ids = cur_log_probs.topk(num_beams)
        cur_word = cur_word.gather(1,topk_ids)
        cur_seqs = torch.cat([cur_seqs[batch_pos,topk_ids],cur_word.unsqueeze(-1)],-1)
        select_index = (batch_beam_pos + beam_coordinate[batch_pos,topk_ids]).view(-1)
        cache = [(k[select_index],v[select_index]) for k,v in cache]
        return end_log_probs_pre,end_flags_pre,end_log_probs,end_seqs,end_flags,cur_log_probs,cur_word,cur_seqs,cache
    
    def grow(cur_log_probs,cur_word,cur_seqs,beam_coordinate,cache):
        cur_log_probs = cur_log_probs[:,:num_beams]
        cur_word = cur_word[:,:num_beams]
        cur_seqs = torch.cat([cur_seqs[batch_pos,beam_coordinate[:,:num_beams]],cur_word.unsqueeze(-1)],-1)
        select_index = (batch_beam_pos + beam_coordinate[:,:num_beams]).view(-1)
        cache = [(k[select_index],v[select_index]) for k,v in cache]
        return cur_log_probs,cur_word,cur_seqs,cache
    
    def final(end_seqs,end_log_probs,cur_seqs,cur_log_probs):
        combine_seqs =  torch.cat([end_seqs, cur_seqs], 1)
        combine_log_probs =torch.cat([end_log_probs, cur_log_probs],1)
        final_log_probs, topk_ids = combine_log_probs.topk(num_beams)
        final_seqs = combine_seqs[batch_pos,topk_ids]
        return final_seqs,final_log_probs
    
    with torch.no_grad():
        cur_log_probs,cur_seqs,cur_word,cache,end_log_probs,end_seqs,end_word,end_flags,end_log_probs_pre,end_flags_pre,vocab_size,batch_size,batch_pos,batch_pos_,batch_beam_pos,length,position_id,left_pad = prepare(input_ids)
        while position_id < max_length - 1 and not stop(cur_log_probs,end_log_probs_pre,end_flags_pre):
            cache,cur_log_probs,cur_word,beam_coordinate,end_seqs,end = step(cur_word,position_id,cache,cur_log_probs,end_seqs)
            if torch.any(end):
                end_log_probs_pre,end_flags_pre,end_log_probs,end_seqs,end_flags,cur_log_probs,cur_word,cur_seqs,cache = update(end_log_probs,end_flags,end,end_seqs,cur_seqs,beam_coordinate,cur_word,cur_log_probs,cache)
            else:
                cur_log_probs,cur_word,cur_seqs,cache = grow(cur_log_probs,cur_word,cur_seqs,beam_coordinate,cache)
            position_id += 1
            length += 1
        return final(end_seqs,end_log_probs,cur_seqs,cur_log_probs)
        
