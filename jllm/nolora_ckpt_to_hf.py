import os
import transformers
from transformers import AutoConfig 
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from .model import ModelPipe
import torch
import tqdm
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="model path")
    parser.add_argument('--ckpt', type=str, help="checkpoint.")
    parser.add_argument('--tag', type=str, default=None, help="checkpoint tag")
    parser.add_argument('--hf',type=str,default="output",help="Where to store the model.")
    args = parser.parse_args()
    os.makedirs(args.hf,exist_ok=True)
    if not args.tag: 
        with open(os.path.join(args.ckpt,'latest')) as f: args.tag = f.read().strip()
    ckpt_path =os.path.join(args.ckpt,args.tag)
    files = os.listdir(ckpt_path)
    device = torch.device('cpu')
    meta_datas = [(int(f.split('_',3)[2]),torch.load(os.path.join(ckpt_path,f), map_location=device)) for f in files if f[:8]=='mp_rank_'] 
    num_stages = len(meta_datas)
    layer_files = [f for f in files if f[:6]=='layer_'] 
    try:
        config = AutoConfig.from_pretrained(args.model,trust_remote_code=True)
    except:
        config = AutoConfig.from_pretrained(args.model)
    config._name_or_path = args.hf
    config.transformers_version = transformers.__version__
    config.to_json_file(os.path.join(args.hf,"config.json"))
    pipe2hf = ModelPipe[config.model_type].get_pipe2hf(config.num_hidden_layers)
    def convert_ckpt2hf(meta_data,layer_files,pipe2hf,num_stages):
        stage_id,meta_data = meta_data
        cur_layer_nums = {int(k.split(".",1)[0]) for dk in meta_data['param_shapes'] for k in dk.keys() }
        cur_layers  = [(int(l[6:].split('-',1)[0]),l) for l in layer_files if int(l[6:].split('-',1)[0]) in cur_layer_nums]
        cur_state_dict ={}
        for i,layer in tqdm.tqdm(cur_layers):
            p = torch.load(os.path.join(ckpt_path,layer), map_location=device)
            for k in list(p.keys()):
                cur_state_dict[pipe2hf[str(i)+'.'+k]] = p.pop(k)
        
        torch.save(cur_state_dict, os.path.join(args.hf,
                     f"pytorch_model-{stage_id + 1:05d}-of-{num_stages:05d}.bin" if num_stages>1 else "pytorch_model.bin"))


    with ProcessPoolExecutor(max_workers=num_stages) as exe:
        func = partial(convert_ckpt2hf,layer_files = layer_files,pipe2hf=pipe2hf,num_stages=num_stages)
        list(exe.map(func,meta_datas))