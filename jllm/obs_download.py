import os
import numpy as np
import argparse
import moxing as mox
from functools import partial
from concurrent.futures import ProcessPoolExecutor
# import torch
# import torch_npu
# from datetime import timedelta
# from torch_npu.contrib import transfer_to_npu

def obs_download(rank,world_size,pp,tp,model,data):
    
    topo=np.arange(world_size).reshape(-1,pp,tp)
    dr,pr,tr=np.where(topo==rank)
    mr=pr*tp+tr
    
    opt= 'bf16_zero_pp_rank_{dr}_mp_rank_{mr:02d}_optim_states.pt'
    md = 'layer_{pr:02d}-model_{tr:02d}-model_states.pt'
    ms = 'mp_rank_{mr:02d}_model_states.pt'
    
    downloads = [
        opt.format(dr=dr.item(),mr=mr.item()),
        md.format(pr=pr.item(),tr=tr.item()),
        ms.format(mr=mr.item())
    ]
    
    if tp>1:
        tp_st = "tensor-{tr:02d}-of-{tp:02d}-pipeline-{pr:02d}-of-{pp:02d}.safetensors"
        downloads.append(tp_st.format(tr=tr.item()+1,tp=tp,pr=pr.item()+1,pp=pp))
    elif pp>1:
        pp_st = "model-{pr:05d}-of-{pp:05d}.safetensors"
        downloads.append(pp_st.format(pr=pr.item()+1,pp=pp))
    else:
        downloads.append('model.safetensors')
    downloads.append('config.json')
    
    for file in downloads:
        file_path = os.path.join(model,file)
        if mox.file.exists(file_path):
            mox.file.copy(file_path,os.path.join('/cache','model',file))
    
    if data is not None and pr==0 and tr ==0 and mox.file.exists(data):
        mox.file.copy(data,'/cache/data')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pp', type=int,default=1,help='pp size' )
    parser.add_argument('--tp', type=int,default=1,help='tp size' )
    parser.add_argument('--data', type=str,help='train data obs path')
    parser.add_argument('--model', type=str,help='model obs path')
    parser.add_argument('--timeout', type=int, default=30, help='timeout')
    args = parser.parse_args()
    
    # init_process_group_kwargs = {
        # 'rank' : int(os.environ.get("RANK", 0)),
        # 'world_size' : int(os.environ.get("WORLD_SIZE", 1)),
        # 'backend' : 'hccl',
        # 'timeout': timedelta(minutes=args.timeout),
        # }

    # torch.distributed.init_process_group(**init_process_group_kwargs)
    # world_size = torch.distributed.get_world_size()
    # rank = torch.distributed.get_rank()
    
    #obs_download(rank,world_size,args.pp,args.tp,args.model,args.data)
    
    with ProcessPoolExecutor(max_workers=8) as exe:
        func = partial(obs_download,
                       world_size=int(os.environ["WORLD_SIZE"]),
                       pp=args.pp,
                       tp=args.tp,
                       model=args.model,
                       data=args.data)
        NODE_RANK = int(os.environ["NODE_RANK"])*8
        list(exe.map(func,range(NODE_RANK,NODE_RANK+8)))
    #torch.distributed.barrier()
    