


# date
# hostname
# file_prefix=$((SGE_TASK_ID - 1))
# source /SAN/orengolab/plm_embeds/cache/miniconda3/miniconda3/bin/activate foldtoken

# export PROFAM_DATA_DIR=/SAN/orengolab/cath_plm/ProFam/data
# 1. remeber to download the FoldToken model weight first from https://zenodo.org/records/13901445, and save to src/tools/foldtoken
ROOT_DIR='/SAN/orengolab/cath_plm/ProFam/profam'
wget https://zenodo.org/records/13901445/files/model_zoom.zip -P $ROOT_DIR/src/tools/foldtoken
unzip $ROOT_DIR/src/tools/foldtoken/model_zoom.zip -d $ROOT_DIR/src/tools/foldtoken
rm -rf $ROOT_DIR/src/tools/foldtoken/__MACOSX
rm $ROOT_DIR/src/tools/foldtoken/model_zoom.zip

# 2. add one line by the end of src/tools/foldtoken/model_zoom/FT4/config.yaml: "k_neighbors: 30"
echo "k_neighbors: 30" >> $ROOT_DIR/src/tools/foldtoken/model_zoom/FT4/config.yaml

# 3. run the programme:
# output_file="/SAN/orengolab/cath_plm/ProFam/data/foldseek_struct/${file_prefix}.parquet"
# if [ ! -f $output_file ]; then
#     echo "Output file not found: $output_file"
#     SCRATCH_DIR=/scratch0/$USER/$JOB_ID/$SGE_TASK_ID
#     mkdir -p ${SCRATCH_DIR}/data
#     echo "Created scratch dir"
#     ls /scratch0/$USER/$JOB_ID
#     export PATH=/SAN/orengolab/cath_plm/ProFam/foldmason/bin/:$PATH
#     # foldmason can in theory scale to very large alignments (in paper they align clusters of size ~ 100000)
#     python3 -m data_creation_scripts.foldseek.create_foldseek_struct_from_db ${SCRATCH_DIR}/data --skip_af50 --minimum_foldseek_cluster_size 1 --parquet_ids $file_prefix --run_foldtoken --foldtoken_level 8 --max_cluster_size_for_foldmason 10000
#     rm -rf ${SCRATCH_DIR}/data/
# fi


#!/bin/bash
#$ -l tmem=12G
#$ -l h_vmem=12G
#$ -l h_rt=38:55:30
#$ -S /bin/bash
#$ -t 1-10000  # 8000 for 2000000 at 250; 1800/4500 for 450000 at 250/100
#$ -N foldseekF
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd ~/ProFam/ahh/profam
#$ -tc 1000
#$ -j y
#$ -l avx2=yes  # for foldmason
##$ -l tscratch=5G
