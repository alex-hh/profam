"""Raw iteration speed.

The most important factor is to avoid conversions between python lists
and numpy arrays. This requires careful handling of formatting etc.

Observations:
- serialising as array3d is helpful for a ~1-1.5x speedup maybe
- setting data format to numpy (or torch) is critical.
- interleaving doesn't affect performance
- in order to apply map efficiently, we needed to fix the format -> map pathway in datasets.

Relevant issues:
- Lazy mapping FR: https://github.com/huggingface/datasets/issues/6012
- Chaining transform with format https://github.com/huggingface/datasets/issues/5910

Solutions:
* write our own interleave datasets function, and wrap hf datasets to handle mapping
* fix hf dataset mapping
* write a function wrapper that wraps any mapped function in a transform function.

TODO: what would be performance difference if we applied filter / map before interleaving?
"""
import argparse
import cProfile
import io
import os
import pstats
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
from src.data.datasets import ProteinDatasetConfig, wrapped_preprocess
from src.data.processors.preprocessing import (
    ParquetStructurePreprocessor,
    PreprocessingConfig,
)
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
    run_dataloader: bool = False,
    null_filter: bool = False,
):
    pr = cProfile.Profile()
    pr.enable()

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
    print("Initial formatting", dataset._formatting, dataset.info.features)
    if format is not None:
        dataset = dataset.with_format(format)

    if null_filter:
        dataset = dataset.filter(lambda x: True)

    def null_map(x):
        return x

    # dataset.info.features = None  can help if filtering is causing a slowdown on old versions of datasets

    preprocess = preprocess_map or preprocess_manual
    if preprocess_null_map:
        # map creates a RebatchedArrowExamplesIterable with batch size batch size
        # a possible cause of problems is ex_iterable iterator converting to python
        dataset = dataset.map(
            null_map,
            batch_size=map_batch_size,
            batched=True if map_batch_size is not None else False,
            format_outputs=False,
            features=dataset.info.features,  # needs to be set when using interleave_datasets
        )

    elif preprocess:
        cfg = ProteinDatasetConfig(
            identifier_col="fam_id",
            preprocessor=ParquetStructurePreprocessor(
                config=PreprocessingConfig(),
                batched_map=True if map_batch_size is not None else False,
                map_batch_size=map_batch_size,
            ),
        )
        tokenizer = ProFamTokenizer(
            tokenizer_file=os.path.join(BASEDIR, "data/profam_tokenizer.json"),
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
        preprocess_fn = wrapped_preprocess(
            preprocess_fn=cfg.preprocessor.preprocess_protein_data,
            cfg=cfg,
            tokenizer=tokenizer,
            dataset_name="foldseek_struct",
            max_tokens_per_example=max_tokens,
            shuffle=True,
        )
        if preprocess_map:
            features = Features(
                **{f: TOKENIZED_FEATURE_TYPES[f] for f in ALL_FEATURE_NAMES}
            )
            # Does applying Map override formatting? that could be one issue...
            dataset = dataset.map(
                preprocess_fn,
                batch_size=map_batch_size,
                remove_columns=[
                    c for c in dataset.column_names if c not in ALL_FEATURE_NAMES
                ],
                batched=True if map_batch_size is not None else False,
                features=features,  # must be set when using interleave_datasets
            )

    if interleave_n > 1:
        dataset = interleave_datasets([dataset] * interleave_n)
        if preprocess_null_map or preprocess_manual or preprocess_null_manual:
            # if preprocess_map, we've already handled formatting within map itself
            dataset = dataset.with_format(
                format
            )  # interleave datasets ignores underlying format(s)

    t1 = time.time()
    print(f"Time to load dataset: {t1 - t0:.4f} seconds")
    print(dataset.info.features)
    if not run_dataloader:
        iterator = iter(dataset)
        for ix, datapoint in enumerate(iterator):
            if preprocess_null_manual:
                datapoint = null_map(datapoint)
            elif preprocess_manual:
                datapoint = preprocess_fn(datapoint)
                datapoint = {
                    k: v for k, v in datapoint.items() if k in ALL_FEATURE_NAMES
                }
            if ix % 100 == 0:
                print(f"datapoint {ix}", {k: type(v) for k, v in datapoint.items()})
            if ix >= max_iters:
                break

        t2 = time.time()
        print(f"Total iteration time: {t2 - t1:.4f} seconds")

    else:
        collator = DocumentBatchCollator(tokenizer=tokenizer)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=loader_batch_size, num_workers=0, collate_fn=collator
        )
        for ix, batch in enumerate(loader):
            if ix * loader_batch_size >= max_iters:
                break

        t2 = time.time()
        print(f"Total iteration time with loader: {t2 - t1:.4f} seconds")

    pr.disable()

    # Print profiling results
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(20)
    print(s.getvalue())


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
    parser.add_argument("--run_dataloader", action="store_true")
    parser.add_argument("--null_filter", action="store_true")
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
        run_dataloader=args.run_dataloader,
        null_filter=args.null_filter,
    )
