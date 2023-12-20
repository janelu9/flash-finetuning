import os
import gc
from concurrent.futures import ProcessPoolExecutor
import torch
import tqdm
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--ckpt', type=str, help="checkpoint.")
    parser.add_argument('-h','--hf',type=str,default="",help="Where to store the model.")
    args = parser.parse_args()
    args.hf = (args.ckpt+"_hf") if not args.hf else args.hf
    os.makedirs(args.hf,exist_ok=True)
    ckpt_path = args.ckpt
    files = os.listdir(ckpt_path)
    device = torch.device('cpu')
    
    meta_datas = [f for f in files if f[:6]=='tensor' and f[7] == '0']
    num_stages = len(meta_datas)

    def func(meta_data):
        pipe_rank = int(meta_data[23:26])
        pts = [(int(f[7]),torch.load(os.path.join(ckpt_path,f), map_location=device)) for f in files if f[8:] == meta_data[8:]]
        pts.sort(key = lambda x:x[0])
        
        keys = list(pts[0][1].keys())
        cur_state_dict ={}
        
        for k in keys:
            if 'input_layernorm' in k:
                q_local_dim = pts[0][1][k].numel()//len(pts)
                break
                
        for k in tqdm.tqdm(keys):
            if "gate_proj" in k or "up_proj" in k or "lm_head" in k:
                state_dict[k] = torch.cat([p[1].pop(k) for p in pts],1)
            elif "embed_tokens" in k or "o_proj" in k or "down_proj" in k:
                state_dict[k] = torch.cat([p[1].pop(k) for p in pts])
            elif "qkv_proj" in k:
                kv_local_dim = (pts[0][1][k].shape[-1] - q_local_dim)//2
                qkvs =[p[1].pop(k).split([q_local_dim,kv_local_dim,kv_local_dim],1) for p in pts]
                state_dict[k.replace("qkv_proj","q_proj")] = torch.cat([q[0] for q in qkvs],1)
                state_dict[k.replace("qkv_proj","k_proj")] = torch.cat([k[1] for k in qkvs],1)
                state_dict[k.replace("qkv_proj","v_proj")] = torch.cat([v[2] for v in qkvs],1)
            else:
                state_dict[k] = pts[0][1].pop(k)
        del pts
        gc.collect()
        
        torch.save(cur_state_dict, os.path.join(args.hf,
                     f"pytorch_model-{pipe_rank:05d}-of-{num_stages:05d}.bin" if num_stages>1 else "pytorch_model.bin"))

    with ProcessPoolExecutor(max_workers=min(num_stages,32)) as exe:
        [i for i in tqdm.tqdm(exe.map(func,meta_datas))]
    print("Done!")