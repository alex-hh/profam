# Integrating a New Single-Sequence Map-Style Dataset into the Existing Pipeline

To integrate the new `ProteinDataset` class (from `single_seq_dataset.py`) into the existing data pipeline, we need to carefully adjust various components to handle single-sequence documents while reusing as much of the existing infrastructure as possible.

Below is an outline of the necessary changes, with specific references to modules, classes, and functions involved.

## Overview

- **Goal**: Integrate a map-style dataset where each example consists of a single protein sequence, using the existing data pipeline designed for multi-sequence documents.
- **Challenges**:
  - Existing preprocessors and tokenizers expect multiple sequences per document.
  - The pipeline uses Hugging Face's `datasets` library with iterable datasets, while the new dataset is a PyTorch `Dataset`.

## Required Changes

1. **Adjust the Preprocessors to Handle Single Sequences**
2. **Modify the Data Loading to Accept PyTorch Datasets**
3. **Update the Tokenizer to Handle Single Sequences**
4. **Adapt the Batch Collator for Single-Sequence Batches**
5. **Integrate the Dataset into a DataModule**
6. **Update Configuration Files**
7. **Ensure Consistency in Data Flow**

Let's delve into each of these steps.

---

## 1. Adjust the Preprocessors to Handle Single Sequences

### Current Situation

- **Preprocessors** (`src/data/preprocessing.py`) like `FastaPreprocessor` and `ParquetSequencePreprocessor` expect examples with multiple sequences, typically in the `sequences` list.
- These preprocessors convert raw data into `ProteinDocument` instances with multiple sequences.

### Required Changes

- **Create a New Preprocessor or Modify Existing Ones**:

  - **Option 1**: Modify the existing preprocessors to handle both single-sequence and multi-sequence cases.
    - Adjust methods to check if the input contains a single sequence.
    - If only a single sequence is present, wrap it in a list to maintain consistency.
  
  - **Option 2**: Create a new preprocessor specifically for single-sequence data.
    - For example, `SingleSequencePreprocessor` inheriting from `BasePreprocessor`.

- **Implementation**:

  ```python
  # src/data/preprocessing.py
  class SingleSequencePreprocessor(BasePreprocessor):
      def preprocess_protein_data(self, example, tokenizer, max_tokens, shuffle):
          sequence = example['sequence']
          sequences = [sequence]  # Wrap in a list to create a single-sequence document
          protein_document = ProteinDocument(
              sequences=sequences,
              # Other fields can be None or populated if available
          )
          # Proceed with tokenization and other preprocessing steps
          # ...
          return tokenized_data
  ```

- **Adjust Conditionals**: Ensure that any logic checking for multiple sequences can handle single-sequence cases gracefully.

---

## 2. Modify the Data Loading to Accept PyTorch Datasets

### Current Situation

- The pipeline uses Hugging Face's `datasets` library for data loading, which expects datasets to be compatible with its framework.
- The new `ProteinDataset` from `single_seq_dataset.py` is a PyTorch `Dataset`, not directly compatible with Hugging Face's datasets.

### Required Changes

- **Option 1: Convert PyTorch Dataset to Hugging Face Dataset**

  - Use the `datasets.Dataset.from_generator` method to convert the PyTorch `Dataset` to a Hugging Face `Dataset`.
  - This allows us to use the existing mapping and preprocessing functions.

  ```python
  from datasets import Dataset

  def dataset_generator():
      for idx in range(len(protein_dataset)):
          yield protein_dataset[idx]

  hf_dataset = Dataset.from_generator(dataset_generator)
  ```

- **Option 2: Adjust the DataModule to Accept PyTorch Datasets**

  - Modify the `ProteinDataMixture` or create a new DataModule to handle PyTorch `Dataset` instances.
  - Ensure that this DataModule applies the necessary preprocessing steps before batching.

- **Implementation Considerations**:

  - If converting to a Hugging Face `Dataset` is not feasible, we need to ensure that the preprocessing functions can be applied to PyTorch datasets.

---

## 3. Update the Tokenizer to Handle Single Sequences

### Current Situation

- `ProFamTokenizer` (`src/utils/tokenizers.py`) expects `ProteinDocument` instances, possibly with multiple sequences concatenated.
- The `encode` method handles sequences with separator tokens and special document tokens.

### Required Changes

- **Modify the `encode` Method**:

  - Ensure the method can handle `ProteinDocument` instances with a single sequence.
  - When concatenating sequences, avoid adding unnecessary separator tokens if there's only one sequence.

- **Implementation**:

  ```python
  # src/utils/tokenizers.py
  class ProFamTokenizer(PreTrainedTokenizerFast):
      def encode(self, proteins: ProteinDocument, document_token="[RAW]", **kwargs):
          sequences = proteins.sequences
          if len(sequences) == 1:
              concatenated_seq = sequences[0]
          else:
              concatenated_seq = self.sep_token.join(sequences)
          # Proceed with encoding
  ```

- **Handle Optional Fields**:

  - Ensure that optional fields like `residue_positions` and `modality_masks` can be `None`.

---

## 4. Adapt the Batch Collator for Single-Sequence Batches

### Current Situation

- `DocumentBatchCollator` (`src/data/utils.py`) collates batches expecting certain tensor shapes based on multiple sequences.

### Required Changes

- **Modify Collation Logic**:

  - Remove or adjust any assumptions about the number of sequences per example.
  - Ensure padding and masking are correctly applied to single-sequence examples.

- **Implementation**:

  ```python
  # src/data/utils.py
  class DocumentBatchCollator:
      def __call__(self, examples):
          # Adjust logic to handle single-sequence examples
          # Use default_collate or custom collation as needed
          # Ensure that 'labels' are correctly assigned
  ```

---

## 5. Integrate the Dataset into a DataModule

### Current Situation

- `ProteinDataMixture` expects datasets following the Hugging Face format and uses the `interleave_datasets` function.

### Required Changes

- **Create a New DataModule or Adjust Existing One**:

  - **Option 1**: Modify `ProteinDataMixture` to accept PyTorch datasets and handle map-style datasets.
  - **Option 2**: Create a new `ProteinSingleSequenceDataModule` specifically for single-sequence datasets.

- **Implementation**:

  ```python
  # src/data/single_sequence_datamodule.py
  class ProteinSingleSequenceDataModule(LightningDataModule):
      def __init__(self, dataset_cfg, tokenizer, **kwargs):
          # Initialize with configurations and tokenizer
          self.dataset_cfg = dataset_cfg
          self.tokenizer = tokenizer
          # Other initialization

      def setup(self, stage: Optional[str] = None):
          # Load the dataset
          self.dataset = ProteinDataset(self.dataset_cfg.fasta_path, tokenizer=self.tokenizer)
          # Apply preprocessing if necessary

      def train_dataloader(self):
          return DataLoader(
              self.dataset,
              batch_size=self.batch_size,
              collate_fn=DocumentBatchCollator(self.tokenizer),
              num_workers=self.num_workers,
          )
  ```

- **Update the Training Script**:

  - In `train.py`, instantiate the new DataModule instead of `ProteinDataMixture`.

---

## 6. Update Configuration Files

### Current Situation

- Configurations are tailored to datasets using pre-defined preprocessors and data paths.

### Required Changes

- **Create a New Dataset Configuration**:

  - For example, `configs/data/dataset/single_sequence.yaml`.

  ```yaml
  defaults:
    - /preprocessor: "single_sequence"
  _target_: src.data.datasets.ProteinDatasetConfig
  data_path_pattern: path/to/single_sequence.fasta
  is_parquet: false
  ```

- **Create a New Preprocessor Configuration**:

  - For example, `configs/preprocessor/single_sequence.yaml`.

  ```yaml
  _target_: src.data.preprocessing.SingleSequencePreprocessor
  ```

- **Update `data.yaml` to Include the New Dataset**:

  ```yaml
  defaults:
    - dataset@dataset_cfgs.single_sequence: single_sequence
    - _self_
  data_weights:
    single_sequence: 1.0
  batch_size: 32
  ```

- **Adjust Paths**: Ensure that `data_dir` and other path variables point to the correct locations.

---

## 7. Ensure Consistency in Data Flow

### Current Situation

- The data flow is designed for multi-sequence documents, with certain expectations in data structures.

### Required Changes

- **Modify the `ProteinDocument` Class if Necessary**:

  - Ensure that it can represent single-sequence documents without issues.
  - Since `sequences` is a list, a single sequence is represented as a list with one element.

- **Adjust Downstream Code**:

  - Ensure that any downstream processing (e.g., in the model) can handle batches where each example contains data from a single sequence.

- **Test the Entire Pipeline**:

  - Validate that data flows correctly from the dataset through preprocessing, tokenization, batching, and into the model.
  - Fix any issues that arise due to changes.

---

## Additional Considerations

- **Handling of Optional Features**:

  - If features like `plddts`, `backbone_coords`, or `structure_tokens` are not available for single sequences, ensure that the code can handle their absence.

- **Data Shuffling and Sampling**:

  - Verify that shuffling and sampling behaviors are appropriate for map-style datasets.

- **Performance and Memory Optimization**:

  - Since map-style datasets load the entire data into memory, ensure that memory usage is acceptable.
  - Consider lazy loading or memory mapping if necessary.

- **Consistency with Existing Codebase**:

  - Maintain consistent coding styles and practices to align with the existing codebase.
  - Reuse existing helper functions and utilities wherever possible.

---

## Example Integration Steps

1. **Implement `SingleSequencePreprocessor` in `src/data/preprocessing.py`**:

   ```python
   # src/data/preprocessing.py
   class SingleSequencePreprocessor(BasePreprocessor):
       def preprocess_protein_data(self, example, tokenizer, max_tokens, shuffle):
           sequence = example['sequence']
           protein_document = ProteinDocument(sequences=[sequence])
           tokenized = tokenizer.encode(protein_document)
           return {'input_ids': tokenized.input_ids, 'labels': tokenized.input_ids}
   ```

2. **Modify `ProteinDataset` in `src/data/single_seq_dataset.py` if Necessary**:

   - Ensure that each example returned matches the expected format for preprocessing.

3. **Create `ProteinSingleSequenceDataModule` in `src/data/single_sequence_datamodule.py`**:

   ```python
   # src/data/single_sequence_datamodule.py
   class ProteinSingleSequenceDataModule(LightningDataModule):
       # Implementation as above
   ```

4. **Update Configuration Files**:

   - `configs/data/dataset/single_sequence.yaml`
   - `configs/preprocessor/single_sequence.yaml`
   - Update `configs/data/data.yaml`

5. **Adjust Training Script (`src/train.py`)**:

   ```python
   # src/train.py
   def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
       tokenizer = hydra.utils.instantiate(cfg.tokenizer)
       datamodule = hydra.utils.instantiate(
           cfg.data,
           tokenizer=tokenizer,
           _target_='src.data.single_sequence_datamodule.ProteinSingleSequenceDataModule'
       )
       model = hydra.utils.instantiate(cfg.model, tokenizer=tokenizer)
       # Continue with training
   ```

6. **Run Tests and Validate**:

   - Ensure that the data pipeline works end-to-end with the new dataset.
   - Validate model training and outputs.

---

## Conclusion

By making targeted modifications to preprocessors, data loading mechanisms, tokenizers, batch collators, and configurations, we can integrate the new `ProteinDataset` class into the existing data pipeline. The key is to ensure compatibility with the expected data structures and to handle cases where the assumptions (e.g., multiple sequences per document) change.

---

## References to Modules and Classes

- **Preprocessors**:

  - **Existing**: `FastaPreprocessor`, `ParquetSequencePreprocessor` in `src/data/preprocessing.py`
  - **New**: `SingleSequencePreprocessor` in `src/data/preprocessing.py`

- **Data Loading**:

  - **Existing**: `load_protein_dataset` in `src/data/custom_datasets.py`
  - **New**: Adapted or new data loading for `ProteinDataset` in `src/data/single_seq_dataset.py`

- **Tokenizers**:

  - **Existing**: `ProFamTokenizer` in `src/utils/tokenizers.py`
  - **Modifications**: Adjust `encode` method to handle single sequences

- **Batch Collator**:

  - **Existing**: `DocumentBatchCollator` in `src/data/utils.py`
  - **Modifications**: Adjust to handle single-sequence batches

- **Data Modules**:

  - **Existing**: `ProteinDataMixture` in `src/data/hf_protein_datamodule.py`
  - **New**: `ProteinSingleSequenceDataModule` in `src/data/single_sequence_datamodule.py`

- **Configuration Files**:

  - **Existing**: `configs/data/data.yaml`, `configs/data/dataset/*.yaml`, `configs/preprocessor/*.yaml`
  - **New**: `configs/data/dataset/single_sequence.yaml`, `configs/preprocessor/single_sequence.yaml`

- **Training Script**:

  - **Existing**: `train.py` in `src/train.py`
  - **Modifications**: Adjust instantiation of DataModule

- **Dataset Class**:

  - **New**: `ProteinDataset` in `src/data/single_seq_dataset.py`

- **Data Structures**:

  - **Existing**: `ProteinDocument` in `src/data/objects.py`

---

By following these steps and making the necessary adjustments, we can successfully integrate the new single-sequence map-style dataset into the existing data pipeline, enabling us to train models on this dataset using the same infrastructure and workflows as before.