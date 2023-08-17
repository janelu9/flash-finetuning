#!/usr/bin/env python
# coding: utf-8
# Created on Thur Jun 29 09:36:49 2023
# @author: Lu Jian
# Email:janelu@live.cn;

from functools import partial
from transformers import LlamaTokenizer
from models.baichuan.tokenization_baichuan import BaichuanTokenizer
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
import pyarrow.parquet
import numpy as np
import argparse
import tqdm
import re
import os
import gc

PROMPT_BEGIN: str = 'BEGINNING OF CONVERSATION: '
PROMPT_USER: str = 'USER: {input} '
PROMPT_ASSISTANT: str = 'ASSISTANT: {answer}'  # should not have a space at the end
PROMPT_INPUT: str = PROMPT_BEGIN + PROMPT_USER
IGNORE_TOKEN_ID: int = -100
MAX_SEQ_LENGTH: int  = 2048
ASSISTANT_LENGTH: int = 5
OVERLAPPING_LENGTH: int  = 128
FILTER_LENGTH: int  = 256
PATTERN: str = "请将下文翻译为英文：{inp}"

def clean_wikitext(string):
    """TODO"""
    return string

def wiki_generator(file,sep="\n\n"):
    with open(file,"r") as f:
        doc=""
        line = f.readline()
        while line:
            doc+=line
            if doc[-2:]==sep:
                p=0;l=len(doc)
                while p<l:
                    yield clean_wikitext(doc[p:p+100000])
                    p += 100000 - 1024
                doc=""
            line = f.readline()  
        if doc:
            yield clean_wikitext(doc)
            
def token_wiki(file,tokenizer):
    for doc in wiki_generator(file):
        ids=tokenizer.encode(doc)
        ids.append(tokenizer.eos_token_id)
        p=0
        n = len(ids)
        while p<n:
            input_ids=ids[p:p+MAX_SEQ_LENGTH]
            l=len(input_ids)
            if FILTER_LENGTH<=l:
                input_ids.extend([tokenizer.pad_token_id]*(MAX_SEQ_LENGTH-l))
                yield {'input_ids':np.array(input_ids, dtype=np.int32)}
            p += MAX_SEQ_LENGTH - OVERLAPPING_LENGTH

def qa_generator(file):
    with open(file,"r") as f:
        line = f.readline()
        while line:
            yield line
            line = f.readline()

def token_qa(file,tokenizer):
    for sample in qa_generator(file):
        inp_anses = sample.strip().split("\t") # example data format: Input\tAnswer, modify by your will.
        offsets=[0]
        inp_ans=inp_anses[:2]
        if len(inp_ans)==2:
            inp,ans=inp_ans
            prompt = PROMPT_INPUT.format(input=PATTERN.format(inp=inp))
            ids=tokenizer.encode(prompt)
            offsets.append(len(ids) + ASSISTANT_LENGTH)
            ads=tokenizer.encode(PROMPT_ASSISTANT.format(answer=ans))[1:]
            ids.extend(ads)
            ids.append(tokenizer.eos_token_id)
            offsets.append(len(ids))
            if offsets[-1] > MAX_SEQ_LENGTH:
                offsets[1] = int((offsets[1] - ASSISTANT_LENGTH)/offsets[-1]*MAX_SEQ_LENGTH)
                ids = ids[:offsets[1]]
                ids.extend(ads[:MAX_SEQ_LENGTH - offsets[1] - 1])
                ids.append(tokenizer.eos_token_id)
                offsets[1] += ASSISTANT_LENGTH
                offsets[-1] = len(ids)
            # for dialog
            i = 2
            while i < len(inp_anses) -1 and offsets[-1] < MAX_SEQ_LENGTH:
                inp_ans=inp_anses[i:i+2]
                if len(inp_ans)==2:
                    inp,ans=inp_ans
                    prompt = PROMPT_USER.format(input=inp)
                    ids.extend(tokenizer.encode(prompt)[1:])
                    offsets.append(len(ids) + ASSISTANT_LENGTH)
                    if offsets[-1] >= MAX_SEQ_LENGTH:
                        break
                    ads=tokenizer.encode(PROMPT_ASSISTANT.format(answer=ans))[1:]
                    ids.extend(ads)
                    ids.append(tokenizer.eos_token_id)
                    offsets.append(len(ids))
                    i += 2
            pad = [tokenizer.pad_token_id]*(MAX_SEQ_LENGTH - offsets[-1])
            ids = ids[:min(MAX_SEQ_LENGTH,offsets[-1])-1]
            ids.append(tokenizer.eos_token_id)
            ids.extend(pad)
            input_ids = np.array(ids, dtype=np.int32)
            labels = input_ids.copy()
            for i in range(0,len(offsets)-1,2):
                s,e=offsets[i:i+2]
                if s<MAX_SEQ_LENGTH:
                    labels[s:e] = IGNORE_TOKEN_ID
            yield {"input_ids":input_ids,"labels":labels}
   
def write_parquet(filename,output_dir,dtype,compression):
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer, fast_tokenizer = True, add_bos_token = True)
    tokenizer.pad_token_id=0
    token = token_wiki
    batch_size = args.batch_size  # size of write batch
    keys = ["input_ids"]
    if dtype == 'qa':
        token = token_qa
        keys.append("labels")
    data_batch={k:[] for k in keys}
    item_iter = token(filename,tokenizer)
    file = os.path.splitext(os.path.basename(filename))[0]
    out_dir = os.path.join(output_dir , file)
    out_file = os.path.join(out_dir , f"{file}-part-%05d.{compression}.parquet")
    check_file = os.path.join(output_dir , "." + file + ".crc")
    if os.path.exists(check_file):
        print(f"{out_file} exists, continue!")
        return
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for i,data in tqdm.tqdm(enumerate(item_iter)):
        for k in data_batch:
            data_batch[k].append(data[k])
        if (i+1) % batch_size == 0 :
            pyarrow.parquet.write_table(pyarrow.table([data_batch[k] for k in keys], names=keys),
                                        out_file % (i//batch_size), 
                                        compression=compression)            
            del data_batch
            gc.collect()
            data_batch={k:[] for k in keys}
    if data_batch[k]:
        pyarrow.parquet.write_table(pyarrow.table([data_batch[k] for k in keys], names=keys),
                                    out_file % (i//batch_size), 
                                    compression=compression)
    del data_batch                                
    os.system(f"echo {i+1} {MAX_SEQ_LENGTH} > {check_file}")
    print(f"{filename} saved with {i+1} samples")
    gc.collect()
    
parser = argparse.ArgumentParser()
parser.add_argument('-t', type=str, default="qa")
parser.add_argument('-i', type=str, default="data.txt")
parser.add_argument('-o', type=str, default="")
parser.add_argument('-n', type=int, default=2**23)
parser.add_argument('-c', type=str, default="gzip",choices=('gzip','brotli','snappy','lz4','zstd'))
parser.add_argument('--batch_size', type=int, default=2**15)
parser.add_argument('--seq_len', type=int, default=2**11)
parser.add_argument('--cores', type=int, default=-1)
parser.add_argument('--tokenizer', type=str, default="openlm-research/open_llama_13b")
parser.add_argument('--tmp', type=str, default="tmp")
parser.add_argument('-T', action='store_true', help="thread")
parser.add_argument('-C', action='store_true', help="clean")
args = parser.parse_args()

if __name__=='__main__':
    print(args)
    MAX_SEQ_LENGTH = args.seq_len
    source=os.path.abspath(args.i)
    source_dir=os.path.dirname(source)
    file =os.path.basename(source)
    file_name = os.path.splitext(file)[0]
    if os.path.isfile(source):
        tmp = os.path.join(source_dir,args.tmp)
        if not os.path.exists(tmp) or args.C:
            os.system(f" rm -rf {tmp}") 
            os.makedirs(tmp)
            os.system(f"cd {tmp};split -d -{args.n} ../{file} {file_name}-part-;cd -;")
    else:
        tmp = source
    compression = args.c.lower()
    output_dir = args.o if args.o !="" else os.path.join(source_dir,file_name+f"_{os.path.basename(args.tokenizer)}")
    if os.path.exists(output_dir):
        if args.C:
            os.system(f" rm -rf {output_dir}/*")
            os.system(f" rm -rf {output_dir}/.*.crc")
    else:
        os.makedirs(output_dir)
    
    Pool = ThreadPoolExecutor if args.T else ProcessPoolExecutor
    cpus=int(os.cpu_count()*0.8) if args.cores <0 else  args.cores
    print(f"########## begine converting {args.t} data with {cpus} executors.###########" )
    with Pool(max_workers=cpus) as exe:
        func = partial(write_parquet,output_dir=output_dir,dtype=args.t,compression=compression)
        files =[os.path.join(tmp, i) for i in os.listdir(tmp)]
        files.sort()
        np.random.shuffle(files)
        list(exe.map(func,files))
    if tmp != source: os.system(f" rm -rf {tmp}") 
    print(f"{source} has been converted into {output_dir} successfully!")
    '''
    Example
    # head -n2 news-commentary-v13-zh-en.txt
    1929年还是1989年?   1929 or 1989?
    巴黎-随着经济危机不断加深和蔓延，整个世界一直在寻找历史上的类似事件希望有助于我们了解目前正在发生的情况。   PARIS – As the economic crisis deepens and widens, the world has been searching for historical analogies to help us understand what has been happening.
    # python convert_raws_to_ids.py -i news-commentary-v13-zh-en.txt -n 30000
    Namespace(t='qa', i='news-commentary-v13-zh-en.txt', o='', n=30000, c='gzip', batch_size=32768, cores=-1, format='parquet', tokenizer='openlm-research/open_llama_13b', tmp='tmp', T=False, C=False)
    /mnt/e/NLP
    ########## begine converting qa data with 12 executors.###########
    12777it [00:06, 2018.59it/s]
    15178it [00:07, 1980.01it/s]/mnt/e/NLP/tmp/news-commentary-v13-zh-en-part-08 saved with 12777 samples
    30000it [00:14, 2070.87it/s]
    30000it [00:14, 2068.94it/s]
    30000it [00:14, 2064.20it/s]
    29999it [00:14, 2062.98it/s]
    30000it [00:14, 2063.26it/s]
    30000it [00:14, 2062.92it/s]
    30000it [00:14, 2056.94it/s]
    30000it [00:14, 2054.50it/s]
    /mnt/e/NLP/tmp/news-commentary-v13-zh-en-part-06 saved with 30000 samples
    /mnt/e/NLP/tmp/news-commentary-v13-zh-en-part-00 saved with 30000 samples
    /mnt/e/NLP/tmp/news-commentary-v13-zh-en-part-05 saved with 30000 samples
    /mnt/e/NLP/tmp/news-commentary-v13-zh-en-part-01 saved with 30000 samples
    /mnt/e/NLP/tmp/news-commentary-v13-zh-en-part-04 saved with 30000 samples
    /mnt/e/NLP/tmp/news-commentary-v13-zh-en-part-02 saved with 29999 samples
    /mnt/e/NLP/tmp/news-commentary-v13-zh-en-part-07 saved with 30000 samples
    /mnt/e/NLP/tmp/news-commentary-v13-zh-en-part-03 saved with 30000 samples
    /mnt/e/NLP/news-commentary-v13-zh-en.txt has been converted into /mnt/e/NLP/news-commentary-v13-zh-en_open_llama_13b successfully!
    '''