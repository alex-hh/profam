import glob
from typing import Any, Dict, Optional
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast, DataCollatorForLanguageModeling
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import bisect
import itertools
import random

from typing import Any, Dict, Optional
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizerFast, DataCollatorForLanguageModeling
from lightning import LightningDataModule
from torch.utils.data import DataLoader
import bisect

import random

def load_protein_dataset(data_path_pattern: str, tokenizer: PreTrainedTokenizerFast,
                         max_tokens: int = 5000, sep_token: str = "[SEP]") -> Dataset:
    def preprocess_fasta(example: Dict[str, Any]) -> Dict[str, Any]:
        sequences = [''.join(one_seq.split('\n')[1:]) for one_seq in example['text'].split('>')[1:]]
        random.shuffle(sequences)
        cumulative_lengths = list(
            itertools.accumulate([len(s) + 1 for s in sequences])
        )  # +1 for separator
        insertion_point = bisect.bisect_left(cumulative_lengths, max_tokens - 2)  # -2 for doc start and end tokens
        concatenated_seqs = sep_token.join(sequences[:insertion_point])
        return tokenizer(concatenated_seqs, truncation=True, max_length=max_tokens,
                         return_tensors="pt", padding="max_length", )  # TODO padding to max length in batch (done in the collator)

    dataset = load_dataset("text", data_files=data_path_pattern, split="train", streaming=True, sample_by='document')
    dataset = dataset.map(preprocess_fasta, batched=False, remove_columns=["text"])
    return dataset

class ProteinDataModule(LightningDataModule):
    def __init__(self, data_path_pattern: str, tokenizer_path: str, batch_size: int = 8, max_tokens: int = 5000):
        super().__init__()
        self.data_path_pattern = data_path_pattern
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.sep_token = "[SEP]"
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_path,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[start-of-document]",
            sep_token=self.sep_token,
            mask_token="[MASK]"
        )
        self.collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)  # TODO add mlm

    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset = load_protein_dataset(self.data_path_pattern, self.tokenizer, self.max_tokens, self.sep_token)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, batch_size=self.batch_size, collate_fn=self.collator)
