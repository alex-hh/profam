import argparse
import os

import datasets
import torch

from src.constants import BASEDIR
from src.data.datasets import ProteinDatasetConfig, load_protein_dataset
from src.data.preprocessing import FastaPreprocessor, PreprocessingConfig
from src.utils.tokenizers import ProFamTokenizer


def test_non_interleaved_shuffle():
    tokenizer = ProFamTokenizer(
        tokenizer_file=os.path.join(
            BASEDIR, "src/data/components/profam_tokenizer.json"
        ),
        mask_token="?",
        sep_token="[SEP]",
        pad_token="[PAD]",
    )
    cfg = ProteinDatasetConfig(
        "ec_example",
        data_path_pattern=os.path.join(BASEDIR, "data/example_data/expasy_ec/*.fasta"),
        preprocessor=FastaPreprocessor(PreprocessingConfig()),
        shuffle=False,
        stream=True,
    )
    data = load_protein_dataset(
        cfg,
        tokenizer=tokenizer,
        max_tokens_per_item=1000,
        shuffle=False,
    )
    data = data.shuffle()
    dataloader = torch.utils.data.DataLoader(
        data,
        batch_size=1,
        num_workers=0,
    )

    for i, batch in enumerate(dataloader):
        print(i, batch["text"][0][:10])

    print("\nNew epoch")
    data = data.set_epoch(1)

    for i, batch in enumerate(dataloader):
        print(i, batch["text"][0][:10])


def test_interleaved_shuffle():
    tokenizer = ProFamTokenizer(
        tokenizer_file=os.path.join(
            BASEDIR, "src/data/components/profam_tokenizer.json"
        ),
        mask_token="?",
        sep_token="[SEP]",
        pad_token="[PAD]",
    )
    cfg = ProteinDatasetConfig(
        "ec_example",
        data_path_pattern=os.path.join(BASEDIR, "data/example_data/expasy_ec/*.fasta"),
        preprocessor=FastaPreprocessor(PreprocessingConfig()),
        shuffle=False,
        stream=True,
    )
    data1 = load_protein_dataset(
        cfg,
        tokenizer=tokenizer,
        max_tokens_per_item=1000,
    )
    data2 = load_protein_dataset(
        cfg,
        tokenizer=tokenizer,
        max_tokens_per_item=1000,
    )
    # interestingly this seems to sample with replacement...
    data = datasets.interleave_datasets(
        [data1, data2], probabilities=[0.5, 0.5], stopping_strategy="all_exhausted"
    )
    data = data.shuffle(buffer_size=1)
    dataloader = torch.utils.data.DataLoader(
        data,
        batch_size=1,
        num_workers=0,
    )

    for i, batch in enumerate(dataloader):
        print(i, batch["text"][0][:10])

    print("\nNew epoch")
    data = data.set_epoch(1)

    for i, batch in enumerate(dataloader):
        print(i, batch["text"][0][:10])


def main(args):
    test_non_interleaved_shuffle()
    test_interleaved_shuffle()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
