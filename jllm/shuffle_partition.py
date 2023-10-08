import pyarrow.parquet
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import tqdm
import time
import os
import gc
import psutil
import argparse

def shuffle_partiton(read_files,output_dir,partition,num_partition = 210,cpus = 1):
    partition_path = os.path.join(output_dir,partition)
    os.makedirs(partition_path,exist_ok = True)
    data = pyarrow.parquet.read_table(read_files)
    data_len=len(data)
    idx = list(range(data_len))
    np.random.shuffle(idx)
    step_len = data_len//num_partition
    remainder = data_len % num_partition
    
    def write_parquets(idx_shuffleds_and_partid):
        idx_shuffled,count = idx_shuffleds_and_partid
        sub_data_sf = data.take(idx_shuffled)
        pyarrow.parquet.write_table(sub_data_sf,
                            os.path.join(partition_path,f"{partition}-part-{count:05}.gzip.parquet"),
                            compression="gzip")
                            
    count = 0
    i = 0
    idx_shuffleds_and_partid = []
    while i<data_len:
        if count<remainder:
            idx_shuffled = idx[i:i+step_len+1]
            i += step_len+1
        else:
            idx_shuffled = idx[i:i+step_len]
            i += step_len
        count += 1
        idx_shuffleds_and_partid.append((idx_shuffled,count-1))
        
    with ThreadPoolExecutor(max_workers=cpus) as exe:
        list(exe.map(write_parquets,idx_shuffleds_and_partid))

    with open(os.path.join(output_dir,f".{partition}.crc"),"w") as f:
        f.write(f"{data_len} {len(data[0][0])} {step_len+1} {len(data.column_names)}\n" )
    
    del data
    gc.collect()
    time.sleep(3)
    
def main(args):
    np.random.seed(args.seed)
    free_mem = psutil.virtual_memory().free
    max_tokens = int(0.8*free_mem/4)
    database = args.database
    output_dir = database +"-sf" if not args.output else args.output
    if os.path.exists(output_dir):
        os.system(f" rm -rf {output_dir}/*")
        os.system(f" rm -rf {output_dir}/.*.crc")
    else:
        os.makedirs(output_dir)
    cpus = max((os.cpu_count()*0.25) if args.cores <0 else args.cores,1)
    tables=[d for d in os.listdir(args.database)if os.path.isdir(os.path.join(args.database,d))]
    for table in tables:
        partitions=[p for p in os.listdir(os.path.join(database,table)) if os.path.isdir(os.path.join(database,table,p))]
        partitions_shape=[list(map(int,open(os.path.join(database,table,f".{f}.crc")).read().split())) for f in partitions]
        partitions_files = [os.listdir(os.path.join(database,table,p)) for p in partitions]
        seq_length = partitions_shape[0][1]
        max_rows = max_tokens//seq_length
        part_id = 0
        num_rows = 0
        read_files = []
        for partition,shape,files in zip(partitions,partitions_shape,partitions_files):
            num_files = len(files)
            data_size,_,batch_size,cols = shape
            for i,file in enumerate(files):
                if i<num_files-1:
                    add_num_rows= batch_size*cols 
                else:
                    add_num_rows= (data_size%batch_size)*cols
                read_files.append(os.path.join(database,table,partition,file))
                num_rows+=add_num_rows
                if num_rows>num_rows:
                    last_file = read_files.pop(-1)
                    shuffle_partiton(read_files,output_dir,f"{table}-part-{part_id:03}",args.num_partition,cpus)
                    part_id += 1
                    num_rows = add_num_rows
                    read_files = [last_file]
        shuffle_partiton(read_files,output_dir,f"{table}-part-{part_id:03}",args.num_partition,cpus)
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--database', type=str, help="parquet database path.")
    parser.add_argument('-o','--output',type=str, default = "",help="output database path.")
    parser.add_argument('-n','--num_partition', type=int, default=210, help="num of split partitions.")
    parser.add_argument('--cores', type=int, default=-1,help="num of task threads.")
    parser.add_argument("--seed",type=int,default=1234, help="A seed for shuffle.")
    args = parser.parse_args()
    main(args)