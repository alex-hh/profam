#$ -l tmem=8G
#$ -l h_vmem=8G
#$ -l h_rt=71:55:30
#$ -S /bin/bash
#$ -N TED7_12
#$ -l hostname=!(burns*)
#$ -t 7-12
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd /SAN/orengolab/cath_plm/ProFam/profam
#$ -j y

# destination="w-judew-1a100v1-41deece431bd432a8eec695a0a230e43@ssh.nvidia-oci.saturnenterprise.io:/home/jovyan/shared/judewells2/profam/data/"
# destination="w-jude1-twoa100pf-2f931b6f99ee47fcb0a4e8fefa909fc8@ssh.nvidia-oci.saturnenterprise.io:/home/jovyan/shared/Jude1B/profamdata/ted/s100_text_min_20_max_90/train_test_split_v2/"
destination="w-jude1-eighta100pf-48b6046641214c58a5ecdd30a0eb7144@ssh.nvidia-oci.saturnenterprise.io:/home/jovyan/shared/Jude1B/profamdata/ted/s100_text_min_20_max_90/train_test_split_v2/"
# destination="w-jude1-eighta100pfv2-6650768276514ccbb9bedc0e6d5b675e@ssh.nvidia-oci.saturnenterprise.io:/home/jovyan/shared/Jude1B/profamdata/ted/s100_text_min_20_max_90/train_test_split_v2/"
#rsync -av /SAN/orengolab/cath_plm/ProFam/data/openfold/uniclust30_clustered_shuffled_final_text $destination
#rsync -av /SAN/orengolab/cath_plm/ProFam/data/foldseek/foldseek_s50_seq_only_text $destination
# rsync -av /SAN/orengolab/cath_plm/ProFam/data/foldseek/foldseek_s100_raw_text $destination
# rsync -av /SAN/orengolab/cath_plm/ProFam/data/funfams $destination
# rsync -av /SAN/orengolab/cath_plm/ProFam/data/uniref/uniref90_text_shuffled $destination
# rsync -av /SAN/orengolab/cath_plm/ProFam/data/ProteinGym $destination
# rsync -av /SAN/orengolab/cath_plm/ProFam/data/foldseek/foldseek_s50_struct/bio2token_tokens $destination
# rsync -av /SAN/orengolab/cath_plm/ProFam/data/foldseek/foldseek_s50_struct/bio2token_parquets_v3 $destination
# rsync -av /SAN/orengolab/cath_plm/ProFam/data/foldseek/foldseek_s50_struct/train_val_test_split_v2 w-judew-1a100v1-41deece431bd432a8eec695a0a230e43@ssh.nvidia-oci.saturnenterprise.io:/home/jovyan/shared/judewells2/profam/data/foldseek/foldseek_s50_struct/

set -euo pipefail

# Source directory to copy from (was previously copied as a whole)
# source_dir="/SAN/orengolab/cath_plm/ProFam/data/ted/s100_text/train_test_split_v2/train_filtered"
source_dir="/SAN/orengolab/cath_plm/ProFam/data/ted/s100_text_min_20_max_90/train_test_split_v2/train_filtered"

# Resolve task indexing
TASK_ID=${SGE_TASK_ID:-1}
TOTAL_TASKS=24

echo "Task ${TASK_ID}/${TOTAL_TASKS} starting on $(hostname)"
echo "Source: ${source_dir}"
echo "Destination: ${destination}"

# Ensure destination directory exists (local or remote)
if [[ "$destination" == *":"* ]]; then
  remote_userhost="${destination%%:*}"
  remote_path="${destination#*:}"
  echo "Ensuring remote directory exists: ${remote_userhost}:${remote_path}"
  ssh "${remote_userhost}" "mkdir -p \"${remote_path}\""
else
  echo "Ensuring local directory exists: ${destination}"
  mkdir -p "${destination}"
fi

# Build list of all source files (relative paths)
file_list=$(mktemp)
assigned_for_task=$(mktemp)

echo "Listing source files..."
( cd "${source_dir}" && find . -type f -print | sed 's|^\./||' ) > "${file_list}"

num_total=$(wc -l < "${file_list}" | tr -d ' ')
echo "Total source files: ${num_total}"

if [[ ${num_total} -eq 0 ]]; then
  echo "No files to copy. Exiting."
  rm -f "${file_list}" "${assigned_for_task}"
  exit 0
fi

# Split work across tasks using round-robin assignment
awk -v m="${TOTAL_TASKS}" -v t="${TASK_ID}" '((NR-1) % m) == (t-1) {print}' "${file_list}" > "${assigned_for_task}"

num_assigned=$(wc -l < "${assigned_for_task}" | tr -d ' ')
if [[ ${num_assigned} -eq 0 ]]; then
  echo "Task ${TASK_ID}: no assigned files. Exiting."
  rm -f "${file_list}" "${assigned_for_task}"
  exit 0
fi

echo "Task ${TASK_ID}/${TOTAL_TASKS}: copying ${num_assigned} files..."

# Transfer assigned files, skipping any that already exist at destination
rsync -av --ignore-existing --files-from="${assigned_for_task}" "${source_dir}/" "${destination}"

rm -f "${file_list}" "${assigned_for_task}"
echo "Task ${TASK_ID}/${TOTAL_TASKS} completed."
