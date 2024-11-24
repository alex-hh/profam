Files in this folder do train/val/test split of TED, FunFam and FoldSeek

run this script to generate the split jsons and split the parquets:
`data_creation_scripts/val_test_split/apply_split_to_parquets.py`
see the following submit script for examples of how to call it:
`submit_scripts/apply_split_to_parquets.qsub.sh`

`apply_split_to_parquets.py` is created in such a way that it will create
all the necessary json split files for foldseek, TED and FunFam datasets
if they have not already been created. Details of the split files below:

splits are based on the ESM-iF paper.
The original split file uses PDB IDs so we need to convert this to CATH 
topology codes eg:
1.20.10 
this is done by:
`make_cath_splits.json`
which creates the following files:
`data/val_test/superfamily_splits.json`
`data/val_test/topology_splits.json`
the latter file can be used directly to split the TED and FunFam datasets 
as they are already identified by CATH codes.

the file `data/val_test/topology_splits.json` is used to create an 
equivalent json splits file for foldseek families with the following
steps:
1) create a json which maps CATH superfamily codes to a list of all uniprot IDs (using TED to assign)
2) any assign a foldseek cluster to "test" if a single UP accession has a topology code in the test set etc.
