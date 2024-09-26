pfam training data msas are extracted from:
https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.full.gz

Run the scripts in the following order:
submit_scripts/
`submit_scripts/grep_pfam_breaks.qsub.sh`
to generate `pfam_end_grepper.txt` 
(containing line-numbers) where each family ends.

Subsequently run:
`submit_scripts/array_job_split_pfam.py`
`data_creation_scripts/pfam/shuffle_pfam_parquets.py`
`data_creation_scripts/pfam/deduplicate_pfam.py`

The following scripts randomly sample familes to be held-out
for evaluation and testing 
Before running the script download the `clustered_split` 
and `random_split` directories from:
[The pfam site](console.cloud.google.com/storage/browser/brain-genomics-public/research/proteins/pfam)
save them in:
`../data/pfam/pfam_eval_splits/`

Note that these split files are splitting sequences WITHIN families.
This is what we use to decide which sequences are used in the prompt
versus the completion sequences. We do our own random selection of families
to hold out from training.

run the following scrips:
`data_creation_scripts/pfam/create_pfam_eval_fastas.py`

Create this file:
`data/val_test/pfam/pfam_val_test_accessions_w_unip_accs.csv`

run this script:
`data_creation_scripts/pfam/train_test_split_pfam_parquets.py`

`train_test_split_pfam_parquets.py` generates a new folder of
parquets in `../data/pfam/train_test_split_parquets`
use this folder for training. 
For validation and testing family-classification use the
fasta files which are saved in:
`data/val_test/pfam/{val/test}/{clustered_split/random_split}/{fam}_{test/train}.fasta`
