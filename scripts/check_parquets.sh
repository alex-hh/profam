#!/bin/bash
# check for existence of parquets with indices in a given range
parquet_dir=$1
max_index=$2
for i in $(seq 0 $max_index);
    do if [ ! -f $parquet_dir/$i.parquet ]; then
        echo "$i"
    fi
done
