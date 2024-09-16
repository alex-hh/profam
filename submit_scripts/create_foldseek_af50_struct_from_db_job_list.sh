#!/bin/bash

job_list_file=$1
date
hostname

while IFS= read -r task_id; do
  # Execute the SUBMIT JOB command (replace it with the actual command)
    file_prefix=$(($task_id - 1))
    qsub ~/profam/submit_scripts/create_foldseek_af50_struct_from_db_job_single.sh $file_prefix
done < $1
