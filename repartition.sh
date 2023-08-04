#! /bin/bash

set -e

cd $1
cur_partitions=`ls`
cur_num=`echo $cur_partitions |wc -w`

if [ $cur_num -gt $2 ] 
then
	partition_num=$2
else
	partition_num=$cur_num
fi

for ((i=0;i<$partition_num ;i++))
do
	file=`printf "part-%05d" $i`
	rm -rf $file
	mkdir $file
done

i=0
for f in `find -name *.parquet|sort|shuf`
do
	d=${f%/*}
	mv $f `printf "part-%05d/${d##*/}-${f##*/}" $((i%$partition_num))`
	i=$((i+1))
done

crc=`ls .*.crc`
cat $crc|awk '{sum+=$1} END {print sum}'>.crc

rm -rf $cur_partitions $crc
