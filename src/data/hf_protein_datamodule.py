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
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import PreTrainedTokenizerFast, DataCollatorForLanguageModeling
from lightning import LightningDataModule
from torch.utils.data import DataLoader
import bisect

import random

def load_protein_dataset(data_path_pattern: str, tokenizer: PreTrainedTokenizerFast,
                         max_tokens: int = 5000, split='train') -> Dataset:
    def preprocess_fasta(example: Dict[str, Any]) -> Dict[str, Any]:
        print(example['text'].split('\n')[0])
        sequences = [''.join(one_seq.split('\n')[1:]) for one_seq in example['text'].split('>')[1:]]
        random.shuffle(sequences)
        cumulative_lengths = list(
            itertools.accumulate([len(s) + 1 for s in sequences])
        )  # +1 for separator
        insertion_point = bisect.bisect_left(cumulative_lengths, max_tokens - 2)  # -2 for doc start and end tokens
        concatenated_seqs = (tokenizer.bos_token + tokenizer.sep_token.join(sequences[:insertion_point])
                             + tokenizer.eos_token)
        return tokenizer(concatenated_seqs, truncation=True, max_length=max_tokens,
                         return_tensors="pt", padding="max_length", add_special_tokens=True)

    dataset = load_dataset("text", data_files=data_path_pattern, split=split, streaming=False, sample_by='document')
    dataset = dataset.map(preprocess_fasta, batched=False, remove_columns=["text"])
    return dataset

class ProteinDataModule(LightningDataModule):
    def __init__(self, data_path_patterns: Dict[str, str],
                 tokenizer_path: str, batch_size: int = 8,
                 max_tokens: int = 5000, num_batches_per_epoch: int = 6):
        super().__init__()
        self.data_path_patterns = data_path_patterns
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.num_batches_per_epoch = num_batches_per_epoch
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_path,
            unk_token="[UNK]",
            pad_token="[PAD]",
            bos_token="[start-of-document]",
            eos_token="[end-of-document]",
            sep_token="[SEP]",
            mask_token="[MASK]",
            add_special_tokens=True
        )
        self.collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)  # TODO add mlm

    def setup(self, stage: Optional[str] = None) -> None:
        datasets = []
        for data_path_pattern in self.data_path_patterns.values():
            dataset = load_protein_dataset(data_path_pattern, self.tokenizer, self.max_tokens, split='train')
            datasets.append(dataset)
        self.train_dataset = concatenate_datasets(datasets)
        self.val_dataset = load_protein_dataset(self.data_path_patterns['ec'],
                                                self.tokenizer, self.max_tokens, split='validation')
        self.test_dataset = load_protein_dataset(self.data_path_patterns['interpro'],
                                                 self.tokenizer, self.max_tokens, split='test')

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.collator, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.datasets['ec'])
    #     p_gym_loader = DataLoader(self.p_gym_val_dataset, batch_size=self.batch_size, collate_fn=self.collator)
    #     cath_loader = DataLoader(self.cath_val_dataset, batch_size=self.batch_size, collate_fn=self.collator)
    #     return [p_gym_loader, cath_loader]
    #
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.datasets['interpro'])
    #     p_gym_loader = DataLoader(self.p_gym_test_dataset, batch_size=self.batch_size, collate_fn=self.collator)
    #     cath_loader = DataLoader(self.cath_test_dataset, batch_size=self.batch_size, collate_fn=self.collator)
    #     return [p_gym_loader, cath_loader]

