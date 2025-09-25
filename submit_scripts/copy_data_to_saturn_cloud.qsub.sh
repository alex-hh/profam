#$ -l tmem=8G
#$ -l h_vmem=8G
#$ -l h_rt=71:55:30
#$ -S /bin/bash
#$ -N copyTED
#$ -t 1
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd /SAN/orengolab/cath_plm/ProFam/profam
#$ -j y

# destination="w-judew-1a100v1-41deece431bd432a8eec695a0a230e43@ssh.nvidia-oci.saturnenterprise.io:/home/jovyan/shared/judewells2/profam/data/"
destination="w-jude1-twoa100pf-2f931b6f99ee47fcb0a4e8fefa909fc8@ssh.nvidia-oci.saturnenterprise.io:/home/jovyan/shared/Jude1B/profamdata/ted/s100_text/train_test_split_v2/"
#rsync -av /SAN/orengolab/cath_plm/ProFam/data/openfold/uniclust30_clustered_shuffled_final_text $destination
#rsync -av /SAN/orengolab/cath_plm/ProFam/data/foldseek/foldseek_s50_seq_only_text $destination
# rsync -av /SAN/orengolab/cath_plm/ProFam/data/foldseek/foldseek_s100_raw_text $destination
# rsync -av /SAN/orengolab/cath_plm/ProFam/data/funfams $destination
# rsync -av /SAN/orengolab/cath_plm/ProFam/data/uniref/uniref90_text_shuffled $destination
# rsync -av /SAN/orengolab/cath_plm/ProFam/data/ProteinGym $destination
# rsync -av /SAN/orengolab/cath_plm/ProFam/data/foldseek/foldseek_s50_struct/bio2token_tokens $destination
# rsync -av /SAN/orengolab/cath_plm/ProFam/data/foldseek/foldseek_s50_struct/bio2token_parquets_v3 $destination
# rsync -av /SAN/orengolab/cath_plm/ProFam/data/foldseek/foldseek_s50_struct/train_val_test_split_v2 w-judew-1a100v1-41deece431bd432a8eec695a0a230e43@ssh.nvidia-oci.saturnenterprise.io:/home/jovyan/shared/judewells2/profam/data/foldseek/foldseek_s50_struct/
rsync -av /SAN/orengolab/cath_plm/ProFam/data/ted/s100_text/train_test_split_v2/train_filtered $destination
