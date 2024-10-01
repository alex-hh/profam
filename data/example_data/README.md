To create foldseek example data:

PROFAM_DATA_DIR=/SAN/orengolab/cath_plm/ProFam/data python3 -m data_creation_scripts.foldseek.create_foldseek_struct_from_db /scratch0/$USER/$JOB_ID --parquet_ids 0 --skip_af50 --run_foldmason --save_dir /SAN/orengolab/cath_plm/ProFam/data/foldseek_struct_example/ --keep_pdbs