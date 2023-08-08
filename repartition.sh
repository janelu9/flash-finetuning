#! /bin/bash

set -e

cd $1
cur_partitions=`ls`
prefix="part"

for ((i=0;i<$2 ;i++))
do
	file=`printf "$prefix-%05d" $i`
	rm -rf $file
	mkdir $file
done

i=0
for f in `find -name *.parquet|sort`
do
	mv $f `printf "$prefix-%05d/" $((i%$2))`
	i=$((i+1))
done

crc=`ls .*.crc`
cat $crc|awk '{sum+=$1} END {print sum}'>.$2.crc

rm -rf $cur_partitions $crc
