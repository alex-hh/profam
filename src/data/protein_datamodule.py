from typing import Any, Dict, Optional
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast, DataCollatorForLanguageModeling
from lightning import LightningDataModule
from torch.utils.data import DataLoader
import bisect
import itertools
import random

class ProteinDataModule(LightningDataModule):
    def __init__(self, data_files: list, tokenizer: str, batch_size: int = 8, max_length: int = 512):
        super().__init__()
        self.data_files = data_files
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = PreTrainedTokenizerFast(
                    tokenizer_file=tokenizer,
                    unk_token="[UNK]",
                    pad_token="[PAD]",
                    cls_token="[start-of-document]",
                    sep_token="[SEP]",
                    mask_token="[MASK]"
                )
        self.collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=True, mlm_probability=0.5)

    

    def setup(self, stage: Optional[str] = None) -> None:
        print(f"Dataset path: {self.data_files}")  # Add this line to display the dataset path
        self.dataset = load_dataset("text", data_files=self.data_files, split="train", streaming=True, sample_by='document')
        # self.dataset = self.dataset.map(self.remove_labels)
        self.dataset = self.dataset.map(self.subsample)
        # self.dataset = self.dataset.map(self.mask_sequences)
        self.dataset = self.dataset.map(self.tokenize, remove_columns=["text"])

    # def remove_labels(self, example: Dict[str, Any]) -> Dict[str, Any]:
    #     example["text"] = "\n".join(
    #         [line for line in example["text"].split("\n") if not line.startswith(">")]
    #     )
    #     return example

    def subsample(self, example: Dict[str, Any], max_length=5000, sequence_end="[SEP]") -> Dict[str, Any]:
        sequences = [
            line for line in example["text"].split("\n") if not line.startswith(">")
        ]
        random.shuffle(sequences)
        cumulative_lengths = list(
            itertools.accumulate([len(s) + 1 for s in sequences])
        )  # +1 for separator
        insertion_point = bisect.bisect_left(cumulative_lengths, max_length)
        example["text"] = sequence_end.join(sequences[:insertion_point]) + sequence_end

        return example

    def tokenize(self, example: Dict[str, Any]) -> Dict[str, Any]:
        return self.tokenizer.batch_encode_plus(
            [example["text"]],  # Ensure this is a list
            add_special_tokens=True,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
    def train_dataloader(self) -> DataLoader:
        for batch in self.dataset:
            print(f"Batch shape: {batch['input_ids'].shape}")
        return DataLoader(self.dataset, batch_size=self.batch_size, collate_fn=self.collator)
        # return DataLoader(self.dataset, batch_size=self.batch_size)

