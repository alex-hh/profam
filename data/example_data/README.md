To create foldseek struct example data:

export PATH=/SAN/orengolab/cath_plm/ProFam/foldmason/bin/:$PATH
PROFAM_DATA_DIR=/SAN/orengolab/cath_plm/ProFam/data python3 -m data_creation_scripts.foldseek.create_foldseek_struct_from_db /scratch0/$USER/$JOB_ID --parquet_ids 0 --skip_af50 --run_foldmason --save_dir /SAN/orengolab/cath_plm/ProFam/data/foldseek_struct_example/ --keep_pdbs

To create foldseek representatives example data:

export PATH=/SAN/orengolab/cath_plm/ProFam/foldseek/bin/:$PATH
PROFAM_DATA_DIR=/SAN/orengolab/cath_plm/ProFam/data python3 -m data_creation_scripts.foldseek.create_foldseek_struct_from_db /scratch0/$USER/$JOB_ID --parquet_ids 0 --skip_af50 --run_foldmason --save_dir /SAN/orengolab/cath_plm/ProFam/data/foldseek_representatives_example/ --keep_pdbs --representative_only