#!/bin/bash
# check for existence of parquets with indices in a given range
parquet_dir=$1
max_index=$2
slurm_job_id=$3

for i in $(seq 0 $max_index);
    do if [ ! -f $parquet_dir/$i.parquet ]; then
        echo "$i"
        if [[ ! -z $slurm_job_id ]]; then
            echo "Logs for failed job $slurm_job_id $(( $i + 1))"
            tail /SAN/orengolab/cath_plm/ProFam/qsub_logs/foldseekF.o${slurm_job_id}.$(( $i + 1 ))
            echo " "
        fi
    fi
done
