#!/bin/bash
# filter ProteinGym MSAs to control maximum identity and minimum coverage (c.f. ProteinNPT)

# The corresponding ProteinNPT command is
# os.system(path_to_hhfilter+os.sep+'bin/hhfilter -cov '+str(hhfilter_min_cov)+' -id '+str(hhfilter_max_seq_id)+' -qid '+str(hhfilter_min_seq_id)+' -i '+preprocessed_filename+'_UC.a2m -o '+output_filename)
# hhfilter_min_cov=75, hhfilter_max_seq_id=90, hhfilter_min_seq_id=0

# N.B. protein gym msa files have this weird focus cols stuff
# what does hhfilter do with that? I think it treats them as inserts.
# we could use reformat.pl with -M first

# Read file into an array, skipping the first line
for msa_file in /SAN/orengolab/cath_plm/ProFam/data/ProteinGym/DMS_msa_files/*.a2m; do
  if [ -e "${msa_file%.a2m}_reformat_hhfilter.a3m" ]; then
    echo "Skipping $msa_file ${msa_file%.a2m} as filtered output exists already"
  else
    echo "Processing $msa_file ${msa_file%.a2m}"
    # convert from the focus col convention to a convention where all cols in the first sequence are considered matches
    reformat.pl -M first $msa_file ${msa_file%.a2m}_reformat.a3m
    hhfilter -cov 75 -id 90 -qid 0 -i ${msa_file%.a2m}_reformat.a3m -o ${msa_file%.a2m}_reformat_hhfilter.a3m
  fi
done
