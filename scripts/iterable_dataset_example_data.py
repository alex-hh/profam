import argparse
import os

import datasets
import torch

from src.constants import BASEDIR
from src.data.builders import HFProteinDatasetConfig, IterableHFProteinDataset
from src.data.processors.preprocessing import (
    PreprocessingConfig,
    ProteinDocumentPreprocessor,
)
from src.data.tokenizers import ProFamTokenizer


def test_non_interleaved_shuffle():
    tokenizer = ProFamTokenizer(
        tokenizer_file=os.path.join(
            BASEDIR, "data/profam_tokenizer.json"
        ),
        mask_token="?",
        sep_token="[SEP]",
        pad_token="[PAD]",
    )
    cfg = HFProteinDatasetConfig(
        data_path_pattern="expasy_ec/*.fasta",
        file_type="text",
    )

    builder = IterableHFProteinDataset(
        name="ec_example",
        cfg=cfg,
        tokenizer=tokenizer,
        preprocessor=ProteinDocumentPreprocessor(PreprocessingConfig()),
    )
    data = builder.load(data_dir=os.path.join(BASEDIR, "data/example_data"))
    data = builder.process(
        data, max_tokens_per_example=1000, shuffle_proteins_in_document=False
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
            BASEDIR, "data/profam_tokenizer.json"
        ),
        mask_token="?",
        sep_token="[SEP]",
        pad_token="[PAD]",
    )
    cfg = HFProteinDatasetConfig(data_path_pattern="expasy_ec/*.fasta")
    builder = IterableHFProteinDataset(
        name="ec_example",
        cfg=cfg,
        tokenizer=tokenizer,
        preprocessor=ProteinDocumentPreprocessor(PreprocessingConfig()),
    )
    data1 = builder.load(data_dir=os.path.join(BASEDIR, "data/example_data"))
    data1 = builder.process(
        data1, max_tokens_per_example=1000, shuffle_proteins_in_document=False
    )
    data1 = data1.shuffle()
    data1 = builder.load(data_dir=os.path.join(BASEDIR, "data/example_data"))
    data2 = builder.load(data_dir=os.path.join(BASEDIR, "data/example_data"))
    data2 = builder.process(
        data2, max_tokens_per_example=1000, shuffle_proteins_in_document=False
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
