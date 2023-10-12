import argparse
from .data import shuffle_datasets

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--datasets', type=str, help="parquet datasets path.")
    parser.add_argument('-o','--output',type=str, default = "",help="output datasets path.")
    parser.add_argument('-n','--num_block', type=int, default=210, help="num of split partitions.")
    parser.add_argument('--cores', type=int, default=-1,help="num of task threads.")
    parser.add_argument('--mem_rate', type=float, default=0.8,help="usage rate of memory.")
    parser.add_argument("--seed",type=int,default=1234, help="A seed for shuffle.")
    args = parser.parse_args()
    shuffle_datasets(args)