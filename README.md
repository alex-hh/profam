<div align="center">

# ProFam: Open-Source Protein Family Language Modeling for Fitness Prediction and Design


</div>

## Description

**ProFam-1** is a 251M-parameter autoregressive protein family language model (pfLM), trained with next-token prediction on **concatenated, unaligned protein sequences** drawn from the same family.

ProFam is built using the **PyTorch Lightning** framework and uses hydra for configuration management.

## Quickstart

### Installation (recommended: `uv`)

If you run into install issues (especially around CUDA / `flash-attn`), jump to [Debugging installation (conda fallback)](#debugging-installation-conda-fallback).

```bash
# clone project
git clone https://github.com/alex-hh/profam.git
cd profam

# create and activate a virtual environment (python 3.11 recommended)
uv venv -p python3.11 .venv
source .venv/bin/activate

# install requirements
uv pip install -r requirements.txt

# (optional) dev tooling
uv pip install -r requirements-dev.txt

# download the model checkpoint
python scripts/hf_download_checkpoint.py
```

### (Recommended) `flash-attn` 2

We **recommend installing FlashAttention 2** it should make (scoring and generating sequences) faster, but these inference pipelines will work fine without it.

If you want to train models using this repo we **strongly** recommend installing Flash Attention as we use **sequence packing** (multiple samples packed with `batch_size=1` and no padding), and this configuration is generally **not supported/without FlashAttention**. To train models without Flash Attention you will need to update the configuration to set `data.pack_to_max_tokens=null`



Install (may require a working CUDA toolchain; see debugging section if it fails):

```bash
uv pip install flash-attn --no-build-isolation
python -c "import flash_attn; print(flash_attn.__version__)"
```

## Repository overview

### Inference entrypoints

There are two main inference scripts:

- **Sampling / generating new sequences**: `scripts/generate_sequences.py`
- **Scoring / log-likelihood**: `scripts/score_sequences.py`

### Training entrypoint

If you want to train ProFam, the entrypoint is:

- **Training**: `src/train.py`

### Input sequence formats (FASTA / MSA)

ProFam can take:

- **Unaligned FASTA** (standard protein sequences), and
- **Aligned / MSA-type files** (e.g. A2M/A3M-style content containing gaps and insertions).

In `scripts/score_sequences.py` we recommend providing an aligned MSA file as we use sequence weighting to encourage sequence diversity when subsampling sequences for the prompt. The weight of a sequence is inversely proportional to the number of similar sequences it has, and this similarity is best computed from an MSA file.

Important: **even if aligned/MSA information is provided, the ProFam model converts these to unaligned gap-free sequences before the forward pass** (i.e. no alignment-aware features are consumed by the model in the standard configs).

During preprocessing, sequences are standardised:

- **Gaps**: `-` (and alignment gap-like `.`) are removed
- **Insertions / lowercase**: lowercase residues (common in A3M insertions) will be converted to uppercase
- **Non-canonical amino acids**: selenocysteine/pyrrolysine are converted (`U → C`, `O → K`). Any remaining out-of-vocabulary characters will map to `[UNK]` if `allow_unk=true` (otherwise they are rejected).

## Training

### Run a lightweight example (no ProFam-Atlas download)

`configs/experiment/train_profam_example.yaml` is configured to run using data in: `data/train_example`.

```bash
python src/train.py experiment=train_profam_example logger=null_logger
```

### Train with the ProFam-Atlas dataset

Training data for ProFam can be downloaded from:

- [Zenodo: ProFam Atlas Dataset](https://zenodo.org/records/17713590)

The default configuration (`configs/train.yaml`) is compatible with the latest ProFam-Atlas release. To run it:

```bash
python src/train.py
```


## Debugging installation (conda fallback)

If the `uv` / `pip` install worked for core dependencies but `flash-attn` fails to build (common when the CUDA toolchain isn’t available), this conda-based approach is a good fallback.

```bash
conda create -n pfenv python=3.11 -y
conda activate pfenv

conda install -c conda-forge ninja packaging -y
conda install -c nvidia cuda-toolkit=12.4 -y

pip install -r requirements.txt

# install a CUDA-enabled PyTorch build (adjust CUDA version/index-url to match your setup)
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121

pip install flash-attn==2.5.6 --no-build-isolation

python -c "import flash_attn; print(flash_attn.__version__)"
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

ProFam uses **text memmap datasets**
for fast random access over large corpora:

- `src/data/text_memmap_datasets.py`: generic **memory-mapped** line access + index building (`*.idx.{npy,info}`)
- `src/data/builders/family_text_memmap_datasets.py`: ProFam-Atlas-specific datasets built on top of the memmap layer

#### ProFam-Atlas on-disk format (`.mapping` / `.sequences`)

The ProFam-Atlas dataset is distributed as paired files:

- **`*.mapping`**: family id + indices into one or more `*.sequences` files
  - **Format**:
    - Line 1: `>FAMILY_ID`
    - Line 2+: `sequences_filename:idx0,idx1,idx2,...`
  - **Important**: `*.mapping` files **must not** have a trailing newline at end-of-file.
- **`*.sequences`**: FASTA-like accessions + sequences
  - **Format** (repeated):
    - `>ACCESSION ...`
    - `SEQUENCE`
  - **Important**: `*.sequences` files **should** have a final trailing newline.

See `README_ProFam_atlas.md` for examples and additional details.

#### How it’s loaded

At a high level, training loads one **protein family** at a time by:

1. Reading a family record from `MappingProteinFamilyMemmapDataset` (a memmapped `*.mapping` dataset)
2. Fetching the referenced sequences from `SequencesProteinFamilyMemmapDataset` (memmapped `*.sequences` files)
3. Building a `ProteinDocument` and preprocessing it (see `src/data/processors/preprocessing.py`)
4. Encoding with `ProFamTokenizer` and forming batches (optionally with packing)

#### Converting FASTA → text memmap

If you have a directory of per-family FASTA files and want to create `*.mapping` / `*.sequences` files for training,
see:

- `data_creation_scripts/fasta_to_text_memmap.py`
