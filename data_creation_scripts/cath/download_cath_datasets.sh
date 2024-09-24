#!/bin/bash
data_dir=$1
mkdir -p $data_dir/cath/cath42
mkdir -p $data_dir/cath/cath43

# Download CATH 4.2 splits from Ingraham paper
# https://github.com/jingraham/neurips19-graph-protein-design
wget -P $data_dir/cath/cath42 https://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/chain_set_splits.json
wget -P $data_dir/cath/cath42 https://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/chain_set.jsonl

# Download CATH 4.3 splits from ESM inverse folding paper
# https://github.com/facebookresearch/esm/tree/main/examples/inverse_folding
wget -P $data_dir/cath/cath43 https://dl.fbaipublicfiles.com/fair-esm/data/cath4.3_topologysplit_202206/chain_set.jsonl
wget -P $data_dir/cath/cath43 https://dl.fbaipublicfiles.com/fair-esm/data/cath4.3_topologysplit_202206/splits.json