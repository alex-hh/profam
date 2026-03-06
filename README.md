<div align="center">

<img src="data/profam_logo_grey.png" alt="ProFam logo" width="720" />

# ProFam: Open-Source Protein Family Language Modelling for Fitness Prediction and Design

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/profam.svg)](https://pypi.org/project/profam/)
[![DOI](https://img.shields.io/badge/DOI-10.64898%2F2025.12.19.695431-blue.svg)](https://www.biorxiv.org/content/10.64898/2025.12.19.695431v1)

</div>

ProFam is an open-source toolkit for training, scoring, and generating protein sequences with protein family language models. It packages the **ProFam-1** 251M-parameter pfLM together with open training and inference workflows, a downloadable pretrained checkpoint, and an open dataset release for reproducible experimentation.

The project is designed for both package users and researchers: you can install it as a standard Python package, run packaged CLI entrypoints with `uv`, and use the repository workflows for family-conditioned scoring, generation, and model training. For background and benchmark results, see the [bioRxiv preprint](https://www.biorxiv.org/content/10.64898/2025.12.19.695431v1).

## Highlights

- Score candidate protein sequences with family context.
- Generate new sequences from FASTA or aligned MSA prompts.
- Download and use the pretrained `ProFam-1` checkpoint.
- Train or reproduce workflows with Hydra-configured experiments.
- Work with both aligned inputs and gap-free modelling pipelines.

## Installation

### From PyPI

Install ProFam as a standard Python package:

```bash
uv pip install profam
```

or

```bash
pip install profam
```

### From Source

If you want the full repository workflows, example data, and inference scripts:

```bash
git clone https://github.com/alex-hh/profam.git
cd profam
uv sync
uv run profam-download-checkpoint
```

Optional installs:

- Development tooling: `uv sync --group dev`
- FlashAttention 2: `uv sync --extra flash-attn`

If you run into CUDA or `flash-attn` issues, see [Installation Details](#installation-details).

## Quickstart

### Verify the installed package

```bash
uv run --with profam --no-project -- python -c "import profam; print(profam.__version__)"
```

### Run a lightweight training example

The bundled example config uses the small dataset under `data/train_example`:

```bash
uv run profam-train experiment=train_profam_example logger=null_logger
```

### Download the pretrained checkpoint

```bash
uv run profam-download-checkpoint
```

## Main Workflows

| Workflow | Purpose | Command |
| --- | --- | --- |
| Train | Train a ProFam model with Hydra configs | `uv run profam-train` |
| Example training | Run a lightweight smoke test on example data | `uv run profam-train experiment=train_profam_example logger=null_logger` |
| Model summary | Print a model architecture summary | `uv run profam-model-summary` |
| Download checkpoint | Fetch the pretrained `ProFam-1` checkpoint | `uv run profam-download-checkpoint` |
| Generate sequences | Sample new sequences from family prompts | `uv run profam-generate-sequences ...` |
| Score sequences | Score candidate sequences with family context | `uv run profam-score-sequences ...` |

The packaged CLI now covers the main package entrypoints, including training, checkpoint download, sequence generation, and sequence scoring.

## Input Sequence Formats

ProFam supports:

- **Unaligned FASTA** for standard protein sequence inputs
- **Aligned / MSA-style files** such as A2M/A3M content with gaps and insertions

For `profam-score-sequences`, we recommend providing an aligned MSA file because sequence weighting is used to encourage diversity when subsampling prompt sequences. Even when aligned inputs are provided, the standard ProFam model converts them into unaligned gap-free sequences before the forward pass.

During preprocessing:

- gaps (`-` and alignment-like `.`) are removed
- lowercase insertions are converted to uppercase
- `U -> C` and `O -> K`
- remaining out-of-vocabulary characters map to `[UNK]` only when `allow_unk=true`

## Training

### Run a lightweight example

`configs/experiment/train_profam_example.yaml` is configured to run on the bundled example data:

```bash
uv run profam-train experiment=train_profam_example logger=null_logger
```

### Train with the ProFam-Atlas dataset

Training data for ProFam can be downloaded from:

- [Zenodo: ProFam Atlas Dataset](https://zenodo.org/records/17713590)

The default configuration in `configs/train.yaml` is compatible with the latest ProFam-Atlas release:

```bash
uv run profam-train
```

## Resources

- [bioRxiv preprint](https://www.biorxiv.org/content/10.64898/2025.12.19.695431v1)
- [Hugging Face: ProFam-1 checkpoint](https://huggingface.co/judewells/ProFam-1)
- [Zenodo: ProFam Atlas Dataset](https://zenodo.org/records/17713590)
- [GitHub repository](https://github.com/alex-hh/profam)

## Citation

If you use ProFam in your work, please cite the preprint:

```bibtex
@article{wells2025profam,
  title = {ProFam: Open-Source Protein Family Language Modelling for Fitness Prediction and Design},
  author = {Wells, Jude and Hawkins Hooker, Alex and Livne, Micha and Lin, Weining and Miller, David and Dallago, Christian and Bordin, Nicola and Paige, Brooks and Rost, Burkhard and Orengo, Christine and Heinzinger, Michael},
  journal = {bioRxiv},
  year = {2025},
  doi = {10.64898/2025.12.19.695431},
  url = {https://www.biorxiv.org/content/10.64898/2025.12.19.695431v1}
}
```

## Installation Details

### CPU-only installation

```bash
uv sync
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### FlashAttention 2

We recommend installing FlashAttention 2 for faster scoring and generation. For training, it is strongly recommended because ProFam uses sequence packing with `batch_size=1` and no padding.

If you need to train without Flash Attention, update the configuration to set `data.pack_to_max_tokens=null`.

```bash
uv sync --extra flash-attn
python -c "import flash_attn; print(flash_attn.__version__)"
```

### Troubleshooting: conda fallback

If a matching `flash-attn` wheel is unavailable and a source build is required, this conda-based fallback is often the easiest route:

```bash
conda create -n pfenv python=3.11 -y
conda activate pfenv

conda install -c conda-forge ninja packaging -y
conda install -c nvidia cuda-toolkit=12.4 -y

pip install profam

# install a CUDA-enabled PyTorch build (adjust CUDA version/index-url to match your setup)
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121

pip install setuptools wheel packaging psutil numpy
pip install flash-attn==2.5.6 --no-build-isolation

python -c "import flash_attn; print(flash_attn.__version__)"
```

## Development

We're using pre-commit to format code and pytest to run tests.

Pull requests will automatically have pre-commit and pytest run on them
and will only be approved once these checks are all passing

Before submitting a pull request, run the checks locally with:

```bash
uv run --group dev pre-commit run --all-files
```

and

```bash
uv run --group dev pytest -k 'not example'
```

Pull requests adding complex new features or making any significant changes
or additions should be accompanied with associated tests in the tests/ directory.

## Concepts

### Data loading

ProFam uses **text memmap datasets**
for fast random access over large corpora:

- `profam/data/text_memmap_datasets.py`: generic **memory-mapped** line access + index building (`*.idx.{npy,info}`)
- `profam/data/builders/family_text_memmap_datasets.py`: ProFam-Atlas-specific datasets built on top of the memmap layer

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
3. Building a `ProteinDocument` and preprocessing it (see `profam/data/processors/preprocessing.py`)
4. Encoding with `ProFamTokenizer` and forming batches (optionally with packing)

#### Converting FASTA → text memmap

If you have a directory of per-family FASTA files and want to create `*.mapping` / `*.sequences` files for training,
see:

- `data_creation_scripts/fasta_to_text_memmap.py`
