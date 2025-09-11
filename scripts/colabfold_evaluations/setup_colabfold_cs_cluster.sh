# 1) Load conda & activate the env
source /SAN/orengolab/cath_plm/localcolabfold/conda/etc/profile.d/conda.sh
conda activate /SAN/orengolab/cath_plm/localcolabfold/colabfold-conda

# 2) Make sure runtime/libs & caches are correct for this cluster
unset PYTHONPATH LD_PRELOAD
export XDG_CACHE_HOME=/SAN/orengolab/cath_plm/.cache
export PIP_CACHE_DIR=/SAN/orengolab/cath_plm/.cache/pip
export TMPDIR=/SAN/orengolab/cath_plm/tmp
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:${LD_LIBRARY_PATH:-}"

# 3) Run
colabfold_batch --help   # or: python -m colabfold.batch --help