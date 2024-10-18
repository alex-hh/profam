pfam training data msas are extracted from:
https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.full.gz

Run the scripts in the following order:
submit_scripts/
`submit_scripts/grep_pfam_breaks.qsub.sh`
to generate `pfam_end_grepper.txt` 
(containing line-numbers) where each family ends.

Subsequently run:
`submit_scripts/array_job_split_pfam.py`

Before running the next script download the `clustered_split` 
and `random_split` directories from:
[The pfam site](console.cloud.google.com/storage/browser/brain-genomics-public/research/proteins/pfam)
save them in:
`../data/pfam/pfam_eval_splits/`

Note that these split files are splitting sequences WITHIN families.
This is what we use to decide which sequences are used in the prompt
versus the completion sequences. We do our own random selection of families
to hold out from training.

run the following scrips:
`data_creation_scripts/pfam/make_pfam_to_up_id.py`
`data_creation_scripts/pfam/consolidated_create_pfam_val_test.py`

See the docstrings in these scripts for further info
