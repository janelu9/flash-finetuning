import os
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--datasets', type=str, help="datasets path.")
    parser.add_argument('-n','--num_partition', type=int, default=16, help="num of partitions.")
    args = parser.parse_args()
    repartition = os.path.join(os.path.dirname(__file__),'repartition.sh')
    os.system(f"chmod 777 {repartition} && {repartition} {args.datasets} {args.num_partition}")