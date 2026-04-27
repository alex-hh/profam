<div align="center">

<img src="data/profam_logo_grey.png" alt="ProFam logo" width="720" />

# ProFam: Open-Source Protein Family Language Modelling for Fitness Prediction and Design

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/profam.svg)](https://pypi.org/project/profam/)
[![DOI](https://img.shields.io/badge/DOI-10.64898%2F2025.12.19.695431-blue.svg)](https://www.biorxiv.org/content/10.64898/2025.12.19.695431v1)

</div>

ProFam is an open-source toolkit for training, scoring, and generating protein sequences with protein family language models. It packages the **ProFam-1** 251M-parameter pfLM together with open training and inference workflows, a downloadable pretrained checkpoint, and an open dataset release for reproducible experimentation.

[bioRxiv preprint](https://www.biorxiv.org/content/10.64898/2025.12.19.695431v1)

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
profam download
```

Optional installs:

- Development tooling: `uv sync --group dev`
- FlashAttention 2: `uv sync --extra flash-attn`

If you run into CUDA or `flash-attn` issues, see [Installation Details](#installation-details).

## Quickstart

### Verify the installed package

```bash
python -c "from profam import ProFam; print('ProFam ready')"
```

### Download the pretrained model weights

The ProFam-1 model weights are hosted on Hugging Face and need to be downloaded before use (or they will be auto-downloaded on first use):

```bash
profam download
```

### Python API

The recommended way to use ProFam programmatically:

```python
from profam import ProFam

model = ProFam()  # loads checkpoint once (auto-downloads if needed)

# Generate sequences conditioned on family context
result = model.generate(
    prompt=["ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIKLMNPQRSTVWF"],
    prompt_accessions=["seq_A", "seq_B"],  # optional: preserved in the result
    num_samples=10,
    top_p=0.95,
)
print(result.sequences)  # list of generated amino acid strings
print(result.scores)     # mean log-likelihood per sequence
# result.conditioning_prompts[i] reports the sequences/accessions that were
# actually fed to the model for ensemble prompt variant i.
for cond in result.conditioning_prompts:
    print(cond.accessions, cond.sequences)

# Score candidate sequences against a family MSA. `prompt` accepts either
# a path (FASTA / a2m / a3m) or an in-memory list[str] of sequences;
# whether the input is an aligned MSA is inferred automatically (every
# sequence must be equal length after stripping a2m/a3m insertions).
# Homology diversity weights are only meaningful for aligned inputs.

# (1) File path to an aligned MSA: diversity weights are available, and
# `cache_weights=True` writes them next to the MSA so subsequent runs skip
# the Hamming computation. `cache_weights=True` requires a file-path prompt.
result = model.score(
    sequences=["ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIKLMNPQRSTVWF"],
    prompt="family.a3m",
    use_diversity_weights=True,  # homology-weighted prompt sub-sampling (default)
    cache_weights=True,          # cache weights next to family.a3m as family_weights.npz
    per_residue=True,            # also return per-position log-likelihoods
)
print(result.scores)          # numpy array of mean log-likelihoods
print(result.residue_scores)  # list[np.ndarray], one per scored sequence

# (2) Aligned in-memory list[str]: '-' represents gaps, lowercase letters
# and '.' are a2m/a3m insertions. Diversity weights are available because
# the sequences are equal-length after stripping insertions. There is no
# source file to cache to, so `cache_weights=True` is rejected here —
# pass a file path (example 1) to cache weights.
result = model.score(
    sequences=["ACDEFGHIKLMNPQRSTVWY"],
    prompt=[
        "ACDEFGHIK-LMNPQRSTVWY",
        "ACDEaFGHIK-LMNPQRSTVWY",  # lowercase 'a' is an a3m-style insertion
        "ACDE-GHIK-LMNPQRSTVWY",
    ],
    use_diversity_weights=True,
)

# (3) Unaligned in-memory list[str]: arbitrary-length sequences. Diversity
# weights are not meaningful here, so pass `use_diversity_weights=False`
# (otherwise a `ValueError` is raised).
result = model.score(
    sequences=["ACDEFGHIKLMNPQRSTVWY"],
    prompt=[
        "ACDEFGHIKLMNPQRSTVWY",
        "ACDEFGHIKLMNPQRSTVW",
        "ACDEFGHIKLMNPQRSTVWYAC",
    ],
    use_diversity_weights=False,
)
```

Homology diversity weights are only meaningful for aligned inputs; with
unaligned input, `use_diversity_weights=True` raises `ValueError`. Weight
caching (`cache_weights=True`) additionally requires a file-path prompt
so the cache file can be keyed to the source MSA — it is not supported
for in-memory `list[str]` prompts.

### CLI

```bash
profam generate --file_path family.fasta --num_samples 10
profam score --conditioning_fasta family.a3m --candidates_file variants.csv
profam download
```

## Main Workflows

| Workflow | Purpose | Command |
| --- | --- | --- |
| Download checkpoint | Fetch the pretrained `ProFam-1` checkpoint | `profam download` |
| Generate sequences | Sample new sequences from family prompts | `profam generate --file_path ...` |
| Score sequences | Score candidate sequences with family context | `profam score --conditioning_fasta ...` |

## Input Sequence Formats

ProFam accepts unaligned FASTA and aligned MSA (A2M / A3M) inputs. Aligned
inputs are preferred for `profam score` so homology-based diversity weights
can be computed. Before the forward pass, the model converts any input to
unaligned gap-free sequences (insertions kept):

- gaps (`-` and alignment-like `.`) are removed
- lowercase insertions are converted to uppercase
- `U -> C` and `O -> K`
- remaining out-of-vocabulary characters map to `[UNK]` only when `allow_unk=true`

## Training

Training is handled via Hydra configs and is intended for development from the source repository (not via pip-installed commands).

### Run a lightweight example

`configs/experiment/train_profam_example.yaml` is configured to run on the bundled example data:

```bash
uv run python -m profam.train experiment=train_profam_example  # requires flash-attn to be installed to support sequence packing
```

### Train with the ProFam-Atlas dataset

Training data for ProFam can be downloaded from:

- [Zenodo: ProFam Atlas Dataset](https://zenodo.org/records/17713590)

The default configuration in `configs/train.yaml` is compatible with the latest ProFam-Atlas release:

```bash
uv run python -m profam.train
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
