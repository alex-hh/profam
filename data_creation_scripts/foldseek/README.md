First run save_pickled_dicts

Next run prepare_job_files. On the cluster the output is saved to afdb/foldseek_job_files
These files contain csvs with information on all proteins within sets of 250 random clusters.

To build parquets from these files, run create_foldseek_struct_from_db with appropriate flags.

Examples of parallelising using the cluster are given in the submit scripts
create_foldseek_af50_struct_from_db_array.qsub.sh
and create_foldseek_representatives_from_db_array.qsub.sh

To handle failures:
~/profam/scripts/check_parquets.sh /SAN/orengolab/cath_plm/ProFam/data/foldseek_af50_struct 9211 > ~/profam_parquets_rerun.txt
bash ~/profam/submit_scripts/create_foldseek_af50_struct_from_db_job_list.sh ~/profam_parquets_rerun.txt

Finally, build index files to help with downstream processing:

e.g.
data_creation_scripts/save_parquet_index.py foldseek_af50_struct