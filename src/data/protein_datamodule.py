from typing import Any, Dict, Optional
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from lightning import LightningDataModule
from torch.utils.data import DataLoader
import bisect
import itertools
import random

class ProteinDataModule(LightningDataModule):
    def __init__(self, data_files: list, batch_size: int = 8, max_length: int = 512):
        super().__init__()
        self.data_files = data_files
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
        self.collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset = load_dataset("text", data_files=self.data_files, split="train", streaming=True)
        self.dataset = self.dataset.map(self.remove_labels)
        self.dataset = self.dataset.map(self.subsample)
        self.dataset = self.dataset.map(self.tokenize, remove_columns=["text"])

    def remove_labels(self, example: Dict[str, Any]) -> Dict[str, Any]:
        example["text"] = "\n".join(
            [line for line in example["text"].split("\n") if not line.startswith(">")]
        )
        return example

    def subsample(self, example: Dict[str, Any], max_length=5000, sequence_separator="\n") -> Dict[str, Any]:
        sequences = [
            line for line in example["text"].split("\n") if not line.startswith(">")
        ]
        random.shuffle(sequences)
        cumulative_lengths = list(
            itertools.accumulate([len(s) + 1 for s in sequences])
        )  # +1 for separator
        insertion_point = bisect.bisect_left(cumulative_lengths, max_length)
        example["text"] = sequence_separator.join(sequences[:insertion_point])
        return example

    def tokenize(self, example: Dict[str, Any]) -> Dict[str, Any]:
        return self.tokenizer(example["text"], truncation=True, padding="max_length", max_length=self.max_length)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, batch_size=self.batch_size, collate_fn=self.collator)