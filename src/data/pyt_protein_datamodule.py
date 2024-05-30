import glob
from typing import Any, Dict, Optional
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast, DataCollatorForLanguageModeling
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import bisect
import itertools
import random

class ProteinDocumentDataset(Dataset):
    def __init__(self, data_path_pattern: str,  tokenizer: PreTrainedTokenizerFast,
                 max_tokens: int = 5000, sep_token: str = "[SEP]"):
        self.sequence_files = glob.glob(data_path_pattern)
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.sep_token = sep_token

    def __len__(self):
        return len(self.sequence_files)

    def preprocess_fasta(self, example: Dict[str, Any]) -> Dict[str, Any]:
        sequences = [''.join(one_seq.split('\n')[1:]) for one_seq in example['text'].split('>')[1:]]
        random.shuffle(sequences)
        cumulative_lengths = list(
            itertools.accumulate([len(s) + 1 for s in sequences])
        )  # +1 for separator
        insertion_point = bisect.bisect_left(cumulative_lengths, self.max_tokens - 2) # -2 for doc start and end tokens
        concatenated_seqs = self.sep_token.join(sequences[:insertion_point])
        return self.tokenizer(concatenated_seqs, truncation=True, max_length=self.max_tokens,
                              return_tensors="pt", padding="max_length")  # TODO padding to max length in batch (done in the collator)


    def __getitem__(self, idx):
        with open(self.sequence_files[idx], 'r') as f:
            fasta_txt = f.read()
        example = {"text": fasta_txt}
        example = self.preprocess_fasta(example)
        return example

class ProteinDataModuleNoHug(LightningDataModule):
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
        self.collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False) # TODO add mlm


    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset = ProteinDocumentDataset(self.data_path_pattern, self.tokenizer, self.max_tokens, self.sep_token)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, batch_size=self.batch_size, collate_fn=self.collator)

