# Detailed AI generated Description of the Data Pipeline

The data pipeline is designed to process protein sequence data from raw formats (such as FASTA and Parquet files) into tokenized inputs suitable for model training and evaluation. The pipeline leverages the Hugging Face `datasets` library for data handling, PyTorch Lightning DataModules for organizing the data loading process, and custom classes and functions for preprocessing and tokenization.

Below is a step-by-step description of the data journey, from raw input files to model-ready batches, with references to the specific modules and classes involved.

## 1. Configuration Setup

### Dataset Configuration

The pipeline begins with configurations defined in YAML files, specifying which datasets to load and how to preprocess them. The main configuration files include:

- **`configs/data/data.yaml`**: Configures the overall data setup, including dataset weights, batch size, and which datasets to include in training and validation.

  ```yaml
  _target_: src.data.hf_protein_datamodule.ProteinDataMixture
  defaults:
    - dataset@dataset_cfgs.openfold_aligned: openfold_aligned
    - dataset@dataset_cfgs.openfold_raw: openfold_raw
    - dataset@dataset_cfgs.ec: ec
    - dataset@dataset_cfgs.gene3dS50: gene3dS50
    - dataset@dataset_cfgs.cathFunfam: cathFunfam
    - dataset@dataset_cfgs.ted: ted
    - dataset/example@dataset_cfgs.interpro: interpro
    - _self_
  data_weights:
    openfold_aligned: 0.125
    openfold_raw: 0.125
    ec: 0.15
    gene3dS50: 0.2
    cathFunfam: 0.1
    ted: 0.3
  batch_size: 22
  max_tokens: 8192
  data_dir: ${paths.data_dir}
  val_dataset_names:
    - interpro
  ```

- **Dataset-Specific Configurations**: Each dataset has its own configuration file, for example:
  - `configs/data/dataset/ec.yaml`
  - `configs/data/dataset/ted.yaml`

  These files specify the data path patterns and assign a preprocessor to each dataset.

  **Example (`ec.yaml`):**

  ```yaml
  defaults:
    - /preprocessor: "fasta_raw"
  _target_: src.data.datasets.ProteinDatasetConfig
  data_path_pattern: ec/ec_fastas/*.fasta
  is_parquet: false
  ```

### Preprocessor Configuration

Preprocessors define how raw data is transformed into a standardized format. Preprocessor configurations are defined in files like:

- `configs/preprocessor/fasta_raw.yaml`
- `configs/preprocessor/fasta_msa.yaml`
- `configs/preprocessor/parquet_raw.yaml`

These files specify which preprocessor class to use for different data formats.

**Example (`fasta_raw.yaml`):**

```yaml
_target_: src.data.preprocessing.FastaPreprocessor
defaults:
  - config: raw
```

## 2. Data Module Initialization

### ProteinDataMixture

The `ProteinDataMixture` class, defined in `src/data/hf_protein_datamodule.py`, is a PyTorch Lightning DataModule that handles loading multiple datasets and mixing them for training.

```python
class ProteinDataMixture(LightningDataModule):
    """Data module for training on a mixture of datasets."""
```

- **Initialization**: It reads the dataset configurations (`ProteinDatasetConfig`) for each dataset and initializes data loaders accordingly.

- **Setup Method**: In the `setup` method, it loads each dataset using the `load_protein_dataset` function and interleaves them according to the specified data weights.

## 3. Dataset Loading

### ProteinDatasetConfig

Each dataset is configured using the `ProteinDatasetConfig` class from `src/data/custom_datasets.py`. It holds information such as data paths, whether to shuffle the data, and which preprocessor to use.

```python
@dataclass
class ProteinDatasetConfig:
    preprocessor: Optional[BasePreprocessor] = None
    data_path_pattern: Optional[str] = None
    is_parquet: bool = False
    shuffle: bool = True
    # Other configuration parameters...
```

### load_protein_dataset Function

The `load_protein_dataset` function, defined in `src/data/custom_datasets.py`, is responsible for loading individual datasets based on the provided configuration.

```python
def load_protein_dataset(
    cfg: ProteinDatasetConfig,
    tokenizer: ProFamTokenizer,
    dataset_name: str,
    # Other parameters...
) -> Dataset:
    # Function body...
```

- **Data Files Preparation**: It uses the `prepare_data_files` helper function to resolve file paths, supporting both glob patterns and explicit file lists.

- **Dataset Loading**:

  - **Parquet Files**: If `is_parquet` is `True`, the dataset is loaded using `load_dataset` with the `"parquet"` format.
  - **FASTA Files**: For FASTA files, the dataset is loaded using `load_dataset` with the `"text"` format, where each document corresponds to a FASTA file.

- **Data Streaming**: The datasets are loaded as streaming datasets if `stream` is `True` in the configuration. This allows processing data on-the-fly without loading the entire dataset into memory.

## 4. Preprocessing and Mapping

### Preprocessors

Preprocessors are responsible for converting raw data into a standardized format suitable for tokenization. Depending on the data format, different preprocessors are used:

- **FastaPreprocessor** (`src/data/preprocessing.py`): Handles raw sequences from FASTA files.

  ```python
  class FastaPreprocessor(BasePreprocessor):
      def preprocess_protein_data(self, example, tokenizer, max_tokens, shuffle):
          # Preprocessing logic for FASTA files...
  ```

- **ParquetSequencePreprocessor** (`src/data/preprocessing.py`): Handles data from Parquet files containing sequences.

  ```python
  class ParquetSequencePreprocessor(BasePreprocessor):
      def preprocess_protein_data(self, example, tokenizer, max_tokens, shuffle):
          # Preprocessing logic for Parquet files...
  ```

### Mapping with wrapped_preprocess

The `wrapped_preprocess` function in `src/data/custom_datasets.py` wraps the preprocessor's `preprocess_protein_data` method and prepares it for use with Hugging Face's `map` function:

```python
def wrapped_preprocess(preprocess_fn, cfg, tokenizer, dataset_name, max_tokens_per_example, shuffle):
    def wrapped_preprocess_fn(example):
        # Calls the preprocessor's preprocess_protein_data method...
        return example
    return wrapped_preprocess_fn
```

- The dataset's `map` method applies this function to each example, transforming it into a standardized format.

### ProteinDocument

During preprocessing, raw data is converted into `ProteinDocument` instances defined in `src/data/objects.py`:

```python
@dataclass
class ProteinDocument:
    sequences: List[str]
    residue_positions: Optional[List[List[int]]] = None
    plddts: Optional[List[np.ndarray]] = None
    backbone_coords: Optional[List[np.ndarray]] = None
    # Other attributes...
```

- **Purpose**: `ProteinDocument` serves as a container for sequence data and associated annotations, such as residue positions and structural information.

## 5. Tokenization

### ProFamTokenizer

The `ProFamTokenizer`, defined in `src/utils/tokenizers.py`, is a custom tokenizer built on top of Hugging Face's `PreTrainedTokenizerFast`.

```python
class ProFamTokenizer(PreTrainedTokenizerFast):
    def encode(self, proteins: ProteinDocument, document_token="[RAW]", padding="max_length", max_length=None, add_final_sep=True, allow_unk=False):
        # Tokenization logic...
        return tokenized
```

- **Encoding**: The tokenizer's `encode` method converts a `ProteinDocument` into token IDs, handling special tokens, padding, and optional embedding of residue positions.

- **Residue Index Embedding**: If `embed_residue_index` is `True`, the tokenizer includes residue position information in the tokenized output.

### Tokenization Process

- The sequences from `ProteinDocument` are concatenated, optionally with separator tokens, document tokens, and BOS/EOS tokens.

- The tokenizer processes the concatenated sequences and generates:

  - **`input_ids`**: Token IDs corresponding to the protein sequences.
  - **Additional Features**: May include `residue_index`, `plddts`, and other annotations.

## 6. Collation into Batches

### DocumentBatchCollator

The `DocumentBatchCollator` in `src/data/utils.py` is responsible for collating individual examples into batches suitable for training.

```python
class DocumentBatchCollator:
    def __init__(self, tokenizer, ignore_gaps=False, feature_names=None):
        # Initialization...
    def __call__(self, examples):
        # Collation logic...
        return batch
```

- **Padding and Masking**: Ensures that sequences are padded to the same length and that padding tokens are properly masked in the labels.

- **Handling Strings**: Manages any string data (such as identifiers) by storing them in custom `StringObject` instances to be carried along in the batch.

### DataLoader

A PyTorch `DataLoader` is created using the prepared dataset and the `DocumentBatchCollator`. This loader yields batches of tokenized data ready for model input.

## 7. Data Module Setup

The `ProteinDataModule` (or `ProteinDataMixture` for multiple datasets) prepares the data loaders for training, validation, and testing.

```python
class ProteinDataMixture(LightningDataModule):
    def setup(self, stage: Optional[str] = None) -> None:
        # Loads and prepares datasets...
    def train_dataloader(self) -> DataLoader:
        # Returns the DataLoader for training...
```

- **Training DataLoader**: Interleaves datasets if multiple are specified, applying data weights to control sampling proportions.

- **Validation and Test DataLoader**: Prepared similarly but usually with different datasets or configurations.

## 8. Model Training Pipeline

### Training Script

The training script (`src/train.py`) ties everything together, initializing the data module, model, and training loop.

```python
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data, tokenizer=tokenizer)
    model: LightningModule = hydra.utils.instantiate(cfg.model, tokenizer=tokenizer)
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
    # Starts training...
```

- **Tokenizer Initialization**: The same `ProFamTokenizer` is used for both data processing and model encoding.

- **Model Initialization**: The model is instantiated with configurations, including the tokenizer.

- **Trainer Initialization**: PyTorch Lightning's `Trainer` manages the training loop, validation, and testing.

## Summary of Data Flow

1. **Configuration Parsing**: YAML configuration files define datasets, preprocessors, tokenizers, and training parameters.

2. **Dataset Loading**:

   - **Raw Data**: Loaded from FASTA or Parquet files using Hugging Face datasets.
   - **Dataset Preparation**: Datasets are prepared according to their configurations.

3. **Preprocessing**:

   - **Per-Example Processing**: Each data example is processed by the appropriate preprocessor, converting raw data into `ProteinDocument` instances.

4. **Tokenization**:

   - **Sequences**: Concatenated and encoded into token IDs by `ProFamTokenizer`.
   - **Annotations**: Residue positions and structural information are included if available.

5. **Batch Collation**:

   - **Padding and Masking**: Batches are created with proper padding and masks using `DocumentBatchCollator`.
   - **Data Loaders**: Data loaders yield these batches for training.

6. **Model Input**:

   - **Training Loop**: Batches are fed into the model during training managed by the PyTorch Lightning `Trainer`.


## References to Modules and Classes

- **Configuration Files**:

  - `configs/data/data.yaml`
  - `configs/data/dataset/*.yaml`
  - `configs/preprocessor/*.yaml`

- **Data Modules**:

  - `ProteinDataMixture` in `src/data/hf_protein_datamodule.py`
  - `ProteinDataModule` in the same module

- **Dataset Configurations**:

  - `ProteinDatasetConfig` in `src/data/custom_datasets.py`

- **Data Loading Functions**:

  - `load_protein_dataset` in `src/data/custom_datasets.py`
  - `prepare_data_files` helper function

- **Preprocessors**:

  - `FastaPreprocessor` in `src/data/preprocessing.py`
  - `ParquetSequencePreprocessor` in the same module

- **Data Structures**:

  - `ProteinDocument` in `src/data/objects.py`

- **Tokenizers**:

  - `ProFamTokenizer` in `src/utils/tokenizers.py`

- **Batch Collator**:

  - `DocumentBatchCollator` in `src/data/utils.py`

- **Training Script**:

  - `train.py` in `src/train.py`

- **Other Utilities**:

  - `StringObject` in `src/data/objects.py`
  - Custom mapping functions in `src/data/custom_datasets.py`

By leveraging these modules and classes, the data pipeline efficiently processes raw protein data into model-ready inputs, enabling training and evaluation of models on complex biological sequences.