
______________________________________________________________________

<div align="center">

# CATH Protein Language Model


</div>

## Description

training a multimodal model on protein documents (TED/AFDB foldseek clusters/openfold MSAs/PFAM...)
[(Docs)](https://docs.google.com/document/d/1UptsPFMFTVyTEu-Ve75NfNpVNrzrWPJlyWvfhi2nsw4/edit)


## Installation

#### Pip

```bash
# clone project
git clone https://github.com/alex-hh/profam.git
cd profam

# [Optional]: create a venv. (Older versions of python also supported currently)
virtualenv -p python3.10 pfenv
source pfenv/bin/activate

# install requirements
pip install -r requirements.txt

# if using flash attention, install separately
pip install flash-attn --no-build-isolation

# if on a development machine, install the follow post-commit hook to track git hash
echo 'git rev-parse HEAD > commit_hash.txt' > .git/hooks/post-commit && chmod +x .git/hooks/post-commit
```
Note 2025-09-25: flash-attn install is no longer working as expected
Requires installing cuda-toolkit (best done in conda) like so:
```
conda create -n pf11 python=3.11
conda activate pf11
conda install -c conda-forge ninja packaging
conda install -c nvidia cuda-toolkit=12.4
pip install -r requirements.txt
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn==2.5.6 --no-build-isolation
```
check it's working:
`python -c "import flash_attn; print(flash_attn.__version__)"`
#### Transferring repo to new filesystems

commit_hash.txt will only be updated locally when commits are made. The post commit hook requires that commit_hash is gitignore. Having installed the git hook one way to upload files to e.g. a cluster that ensures that the commit hash is preserved is therefore to do e.g.:

```bash
rsync -av --max-size=5m --exclude-from='.syncignore' ./ USER@REMOTE_IP:/home/ubuntu/profam-usw/profam
```

#### Loading environment on UCL cs cluster

```bash
source /SAN/orengolab/cath_plm/ProFam/pfenv.source
export PROFAM_DATA_DIR=/SAN/orengolab/cath_plm/ProFam/data  # for what?
```

(The former file is at scripts/pfenv.source)


#### Uploading datasets to the hub

```bash
huggingface-cli upload-large-folder profam/<FOLDER_NAME> <FOLDER_NAME> --repo-type dataset --num-workers <num_cores>
```

#### Loading environment and data on a new cluster

Follow the installation instructions above then:


```bash
huggingface-cli login  # now enter your token, and store as git credential
huggingface-cli download profam/<FOLDER_NAME> --repo-type dataset --local-dir <PATH/TO/DATA/DIR>/<FOLDER_NAME>
# the files should now be at <PATH/TO/DATA/DIR>/<FOLDER_NAME>/*.parquet
# now train/benchmark, setting paths.data_dir appropriately
HYDRA_FULL_ERROR=1 python src/train.py ... paths.data_dir=PATH/TO/DATA/DIR
```


## Introduction

The repo is built on top of [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template).

There are two main entrypoints configured by hydra, for running benchmarks and
training runs given configuration provided in config files.

We favour configuring a given benchmarking run via a config in configs/benchmarks
and a given training run via a config in configs/experiments. See the folders
for examples of config files and below for example commands.


### Benchmarking

```bash
python src/run_sampling_evaluation.py benchmark=<BENCHMARK_CFG_NAME>
```

### Training


Run on example data

```bash
python src/train.py logger=null_logger
```

Train model with default configuration

```bash
# train on CPU / GPU / multiple GPUS
python src/train.py trainer=[cpu/gpu/ddp]
```

Train model with an experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

Override configuration parameters using standard hydra syntax

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```

## Development

We're using pre-commit to format code and pytest to run tests.

Pull requests will automatically have pre-commit and pytest run on them
and will only be approved once these checks are all passing

Before submitting a pull request, run the checks locally with:

```bash
pre-commit run --all-files
```

and

```bash
pytest -k 'not example'
```

Pull requests adding complex new features or making any significant changes
or additions should be accompanied with associated tests in the tests/ directory.


## Concepts

### Data loading

We use iterable HF Dataset instances to load raw data representing protein documents
(e.g. dictionaries of sequences, cluster ids, coords, etc.) from parquet files.

During data loading and benchmarking, raw protein document data is converted into
instances of the ProteinDocument dataclass.

During training, data loading involves a few steps:
  1. A protein document instance is extracted from raw (e.g. parquet/fasta) data
  2. The protein document is preprocessed, with standardisations and other transformations
     applied to its contents (e.g. subsampling of sequences, calculation of MSA-aware sequence
     positions, rotation of backbone coordinates...)
  3. The protein document is encoded into a dictionary of numpy arrays by the ProFamTokenizer
  4. Batches are constructed by collating protein document tensor dictionaries.

Steps 1-3 are handled by a Preprocessor instance, which has configuration determining how
the document should be extracted and processed. Preprocessors live in src/data/preprocessing.py,
and are applied to dataset instances via the hf datasets map() function (which runs on-the-fly
on iterable datasets during training.)

A preprocessor applies a set of transform functions to the document. Transforms are functions
which accept protein documents as input and return a new protein document. These live in
src/data/transforms.py.

### Benchmarking

The benchmarks are organised around three sets of objects:
* pipelines (src/pipelines): represent collections of test documents and handle saving results and running models on these test cases
* evaluators (src/evaluators): basically functions which compute metrics given the input test document and the model's outputs (e.g. generations)
* samplers/scorers: wrappers for profam and baseline models which implement standard methods called by pipeline.
  - e.g. samplers should have a sample_seqs method which accepts as input a ProteinDocument and returns sequences


## Project Directory Structure
```
├── .github                   <- Github Actions workflows
│
├── configs                   <- Hydra configs
│   ├── experiment               <- Configs for training runs
|   |── benchmark                <- Configs for benchmarks
|   .
|   .
│   ├── sampling_eval.yaml    <- Main config for sampling evaluation
│   └── train.yaml            <- Main config for training
│
├── data                   <- Project data
│
├── logs                   <- Logs generated by hydra and lightning loggers
│
├── notebooks              <- Jupyter notebooks.
│
├── scripts                <- Shell scripts
│
├── src                    <- Source code
│   ├── data                     <- Data classes, datasets and associated loading and transformation utils
|   |── evaluators               <- Evaluators compute metrics given model outputs on benchmark instances
|   |── pipelines                <- Pipelines define sets of instances constituting a benchmark, and handle collecting and saving metrics from evaluators
|   |── baselines                <- Wrappers for baselines to make them consistent with our benchmarking API
|   |── sequence                 <- Utilities for processing protein sequences
|   |── structure                <- Utilities for processing protein structures
│   ├── models                   <- Model scripts
|   |── tools                    <- Wrappers for external bioinformatics tools e.g. foldseek / hmmer
│   ├── utils                    <- Utility scripts
│   │
│   ├── run_sampling_evaluation.py <- Run sampling benchmarks
│   └── train.py                   <- Run training
│
├── tests                  <- Tests of any kind
│
├── .gitignore                <- List of files ignored by git
├── .pre-commit-config.yaml   <- Configuration of pre-commit hooks for code formatting
├── .project-root             <- File for inferring the position of project root directory
├── environment.yaml          <- File for installing conda environment
├── pyproject.toml            <- Configuration options for testing and linting
├── requirements.txt          <- File for installing python dependencies
├── setup.py                  <- File for installing project as a package
└── README.md
```

<br>

## Results and analysis for paper
