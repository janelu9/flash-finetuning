#! /bin/bash

set -e

cd $1
cur_partitions=`ls`
uuid=`head -1 /dev/urandom | od -x |head -1| awk '{print $2$3$4$5$6$7$8$9}'`

for ((i=0;i<$2 ;i++))
do
	file=`printf "$uuid-part-%05d" $i`
	rm -rf $file
	mkdir $file
done

i=0
for f in `find -name *.parquet|sort`
do
	mv $f `printf "$uuid-part-%05d/" $((i%$2))`
	i=$((i+1))
done

rm -rf $cur_partitions 
crc=`ls .*.crc`
cat $crc|awk '{sum+=$1} END {print sum}'>.$uuid.crc
rm -f $crc
echo -e "Data: $1\nPartitions: $2\nFiles: $i\nSamples: $(cat .$uuid.crc)\nUUID: $uuid"> data.info
