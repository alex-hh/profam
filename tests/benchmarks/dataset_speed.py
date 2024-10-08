"""Raw iteration speed.

Observations:
- serialising as array3d is helpful for a ~1-1.5x speedup maybe
- setting data format to numpy (or torch) is critical.
- interleaving doesn't affect performance
- in order to apply map efficiently, we need to set the format to arrow first.
    https://github.com/huggingface/datasets/issues/6833
    https://github.com/huggingface/datasets/issues/7206
    however, presumably this requires manually converting to numpy in the map function...

Relevant issues:
- Lazy mapping FR: https://github.com/huggingface/datasets/issues/6012
- Chaining transform with format https://github.com/huggingface/datasets/issues/5910

Solutions:
* write our own interleave datasets function, and wrap hf datasets to handle mapping
* fix hf dataset mapping
* write a function wrapper that wraps any mapped function in a transform function.
"""
import argparse
import os
import time
from typing import Optional

import torch
from datasets import Features, interleave_datasets, load_dataset

from src.constants import (
    ALL_FEATURE_NAMES,
    BASEDIR,
    PROFAM_DATA_DIR,
    TOKENIZED_FEATURE_TYPES,
)
from src.data.preprocessing import ParquetStructurePreprocessor, PreprocessingConfig
from src.data.utils import DocumentBatchCollator
from src.utils.tokenizers import ProFamTokenizer


def main(
    max_iters: int,
    data_folder: str,
    format: Optional[str] = None,
    interleave_n: int = 1,
    loader_batch_size: int = 16,
    max_tokens: int = 2048,
    preprocess_map: bool = False,
    preprocess_manual: bool = False,
    preprocess_null_map: bool = False,
    preprocess_null_manual: bool = False,
    map_batch_size: Optional[int] = None,
):
    t0 = time.time()
    print(
        f"Loading files matching glob: {os.path.join(PROFAM_DATA_DIR, f'{data_folder}/*.parquet')}"
    )
    # can we set batch_size for loader? is it useful
    dataset = load_dataset(
        path="parquet",
        data_files=os.path.join(PROFAM_DATA_DIR, f"{data_folder}/*.parquet"),
        split="train",
        streaming=True,
    )
    print("Initial formatting", dataset._formatting)
    if format is not None:
        dataset = dataset.with_format(format)

    def null_map(x):
        return x

    preprocess = preprocess_map or preprocess_manual
    if preprocess_null_map:
        # map creates a RebatchedArrowExamplesIterable with batch size batch size
        # a possible cause of problems is ex_iterable iterator converting to python
        dataset = dataset.map(
            null_map,
            features=dataset.features,
            batch_size=map_batch_size,
            batched=True if map_batch_size is not None else False,
        )
    elif preprocess:
        tokenizer = ProFamTokenizer(
            tokenizer_file=os.path.join(
                BASEDIR, "src/data/components/profam_tokenizer.json"
            ),
            unk_token="[UNK]",
            pad_token="[PAD]",
            bos_token="[start-of-document]",
            sep_token="[SEP]",
            mask_token="?",
            seq_struct_sep_token="|",
            add_final_sep=True,
            add_bos_token=True,
            add_document_token=True,
        )
        preprocessor = ParquetStructurePreprocessor(
            config=PreprocessingConfig(),
        )
        if preprocess_map:
            features = Features(
                **{f: TOKENIZED_FEATURE_TYPES[f] for f in ALL_FEATURE_NAMES}
            )
            # Does applying Map override formatting? that could be one issue...
            dataset = dataset.map(
                preprocessor.preprocess_protein_data,
                fn_kwargs={"tokenizer": tokenizer, "max_tokens": max_tokens},
                batch_size=map_batch_size,
                features=features,
                remove_columns=[
                    c for c in dataset.column_names if c not in ALL_FEATURE_NAMES
                ],
                batched=True if map_batch_size is not None else False,
            )

    if interleave_n > 1:
        dataset = interleave_datasets([dataset] * interleave_n)
        dataset = dataset.with_format(
            format
        )  # interleave datasets ignores underlying format(s)
    t1 = time.time()
    print(f"Time to load dataset: {t1 - t0:.4f} seconds")
    print(dataset.info.features)
    iterator = iter(dataset)
    for ix, datapoint in enumerate(iterator):
        if preprocess_null_manual:
            datapoint = null_map(datapoint)
        elif preprocess_manual:
            datapoint = preprocessor.preprocess_protein_data(
                datapoint,
                tokenizer=tokenizer,
                max_tokens=max_tokens,
            )
            datapoint = {k: v for k, v in datapoint.items() if k in ALL_FEATURE_NAMES}
        if ix >= max_iters:
            break

    if preprocess_null_manual or preprocess_null_map:
        print(datapoint["plddts"])
    t2 = time.time()
    print(f"Total iteration time: {t2 - t1:.4f} seconds")

    if preprocess_map:
        collator = DocumentBatchCollator(tokenizer=tokenizer)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=loader_batch_size, num_workers=0, collate_fn=collator
        )
        for ix, batch in enumerate(loader):
            if ix * loader_batch_size >= max_iters:
                break

        t3 = time.time()
        print(f"Total iteration time with loader: {t3 - t2:.4f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iters", type=int, default=1000)
    parser.add_argument("--data_folder", type=str, default="foldseek_struct")
    parser.add_argument(
        "--format", type=str, default=None, choices=["numpy", "torch", "arrow"]
    )
    parser.add_argument("--interleave_n", type=int, default=1)
    parser.add_argument("--loader_batch_size", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--map_batch_size", type=int, default=None)
    parser.add_argument("--preprocess_map", action="store_true")
    parser.add_argument("--preprocess_manual", action="store_true")
    parser.add_argument("--preprocess_null_map", action="store_true")
    parser.add_argument("--preprocess_null_manual", action="store_true")
    args = parser.parse_args()
    main(
        args.max_iters,
        args.data_folder,
        format=args.format,
        interleave_n=args.interleave_n,
        loader_batch_size=args.loader_batch_size,
        max_tokens=args.max_tokens,
        preprocess_map=args.preprocess_map,
        preprocess_manual=args.preprocess_manual,
        preprocess_null_map=args.preprocess_null_map,
        preprocess_null_manual=args.preprocess_null_manual,
        map_batch_size=args.map_batch_size,
    )
