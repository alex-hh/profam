#!/bin/bash

job_list_file=$1
date
hostname

while IFS= read -r file_prefix; do
  # Execute the SUBMIT JOB command (replace it with the actual command)
    echo Submitting job to create parquet $file_prefix
    qsub ~/profam/submit_scripts/create_foldseek_af50_struct_from_db_single.qsub.sh $file_prefix
done < $1
