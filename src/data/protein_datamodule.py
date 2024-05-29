from typing import Any, Dict, Optional
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast, DataCollatorForLanguageModeling
from lightning import LightningDataModule
from torch.utils.data import DataLoader
import bisect
import itertools
import random

class ProteinDataModule(LightningDataModule):
    def __init__(self, data_files: list, tokenizer: str, batch_size: int = 8, max_tokens: int = 5000):
        super().__init__()
        self.data_files = data_files
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.tokenizer = PreTrainedTokenizerFast(
                    tokenizer_file=tokenizer,
                    unk_token="[UNK]",
                    pad_token="[PAD]",
                    cls_token="[start-of-document]",
                    sep_token="[end-of-document]",
                    mask_token=None # Add mask token if needed
                )
        self.collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset = load_dataset("text",
                                    data_files=self.data_files,
                                    split="train",
                                    streaming=True,
                                    sample_by='document')
        self.dataset = self.dataset.map(self.preprocess_fasta)
        self.dataset = self.dataset.map(self.subsample)
        self.dataset = self.dataset.map(self.tokenize, remove_columns=["text"])


    def preprocess_fasta(self, example: Dict[str, Any], sequence_separator="\n") -> Dict[str, Any]:  # TODO this method should not have any parameters: sequence separator should be a class attribute consistent with tokenizer
        sequences = [''.join(one_seq.split('\n')[1:]) for one_seq in example['text'].split('>')[1:]]
        example['text'] = sequences
        random.shuffle(sequences)
        cumulative_lengths = list(
            itertools.accumulate([len(s) + 1 for s in sequences])
        )  # +1 for separator
        insertion_point = bisect.bisect_left(cumulative_lengths, self.max_tokens - 2) # -2 for doc start and end tokens
        example["text"] = sequence_separator.join(sequences[:insertion_point])
        return example

    def tokenize(self, example: Dict[str, Any]) -> Dict[str, Any]:
        '''
        input [List] exmaple: ["ARNDC [start-of-document] QEGHIL KMFPST WYV [end-of-document] [PAD] [UNK]", "KMFPST"]
        output [Dist]: {"input_ids": tensor([[]]), "attention_mask": tensor([[]])}
        '''

        return self.tokenizer.batch_encode_plus(example["text"],
                                  add_special_tokens=True,
                                  padding="longest",
                                  return_tensors='pt')

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, batch_size=self.batch_size, collate_fn=self.collator)