# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import bisect
from functools import lru_cache
from typing import Any, List, Optional, Tuple

import numpy as np
import torch

from src.utils import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)

__all__ = [
    "OnlineSampleMapping",
    "OnlineSampleMappingDataset",
    "WeightedConcatOnlineDataset",
]


def handle_index(ds_size: int, idx: int) -> int:
    """
    Remaps negative indices and handles numpy int indices.

    Arguments:
        ds_size (int): Size of the dataset.
        idx (int): Index. Can include negative indices.
    Returns:
        int: Remapped and fully qualified index.

    Raises:
        IndexError: If a negative index is out of range.

    Examples:
        >>> import numpy as np
        >>> handle_index(5, 1)
        1
        >>> handle_index(5, -2)
        3
    """
    if idx < 0 and idx > -ds_size - 1:
        idx = ds_size + idx
    elif idx < 0:
        raise IndexError(f"Index {idx} out of range")
    return idx


class OnlineSampleMapping:
    """
    This class is used to create a sample mapping for certain number of samples, including pseudo-random shuffling.
    The sampler allows to down, or upsample a given dataset.
    Shuffling leads to pseudo-random shuffling, where blocks are shuffled,
    and each block is internally shuffled.
    """

    def __init__(
        self,
        dataset_size: int,
        block_size: Optional[int] = 1000000,
        cache_maxsize: int = 2,
        seed: int = 1,
        shuffle: bool = True,
        truncate_to_block_boundary: bool = False,
    ) -> None:
        """
        Args:
            dataset_size (int): Size of the dataset.
            block_size (Optional[int]): Size of each sample block. Defaults to 1000000.
            cache_maxsize (int): Maximum size of the blocks cache for the get_sample_block function.
            seed (int): Seed for the random number generator used for shuffling.
            shuffle (bool): Whether to shuffle the samples.
            truncate_to_block_boundary (bool): Whether to truncate the last block to the block boundary.
        """
        self.dataset_size = dataset_size
        self.num_samples = dataset_size
        self.block_size = block_size if block_size is not None else self.dataset_size
        self.cache_maxsize = cache_maxsize
        self.seed = seed
        self.shuffle = shuffle
        self.truncate_to_block_boundary = truncate_to_block_boundary
        self.required_samples = max(self.num_samples, self.dataset_size)
        # block size cannot be larger than dataset size
        self.block_size = min(self.block_size, self.dataset_size)
        # reduce the last block if needed, to match the required number of samples
        last_block_size = self.required_samples % self.block_size
        # store required blocks to cover num_samples samples and dataset_size samples
        self.num_blocks = int(np.ceil(self.required_samples / self.block_size))

        # if required, truncate the last block to the block boundary
        if self.truncate_to_block_boundary and last_block_size:
            # update num_samples to account for truncated last block only if needed
            if self.required_samples == self.num_samples:
                self.num_samples -= last_block_size

            # apdate num_blocks to account for truncated last block
            self.num_blocks -= 1
            self.required_samples -= last_block_size
            last_block_size = 0

        # create a list of blocks (should cover the entire dataset for correct down sampling)
        block_idx_list = np.arange(self.num_blocks)
        # compute the size of each block
        block_size_list = np.full(self.num_blocks, self.block_size)
        if last_block_size:
            block_size_list[-1] = last_block_size
            self.use_digitize = True
        else:
            self.use_digitize = False
        if shuffle:
            local_rng = np.random.RandomState(seed=self.seed)
            idx = local_rng.permutation(np.arange(self.num_blocks))
            block_idx_list = block_idx_list[idx]
            block_size_list = block_size_list[idx]

        # store only required number of blocks
        self.block_idx_list = block_idx_list
        self.block_size_list = block_size_list
        self.block_bins = np.cumsum(block_size_list)

        # NOTE: MAKE get_sample_block A CACHED FUNCTION!!!
        self.get_sample_block = lru_cache(maxsize=cache_maxsize, typed=False)(
            self.get_sample_block
        )

    def __str__(self) -> str:
        return (
            f"OnlineSampleMapping("
            f"dataset_size={self.dataset_size}, "
            f"num_samples={self.num_samples}, "
            f"required_samples={self.required_samples}, "
            f"block_size={self.block_size}, "
            f"num_blocks={self.num_blocks}, "
            f"cache_maxsize={self.cache_maxsize}, "
            f"seed={self.seed}, "
            f"shuffle={self.shuffle}, "
            f"truncate_to_block_boundary={self.truncate_to_block_boundary}"
            f")"
        )

    def __getitem__(self, idx: int | slice) -> int | list[int]:
        # handle slices
        if isinstance(idx, slice):
            slc = idx
            start, stop, step = slc.start, slc.stop, slc.step

            # Handle None values
            start = handle_index(self.num_samples, start if start is not None else 0)
            if start >= self.num_samples:
                start = self.num_samples
            stop = handle_index(
                self.num_samples, stop if stop is not None else self.num_samples
            )
            if stop >= self.num_samples:
                stop = self.num_samples
            step = step if step is not None else 1
            sample_slice = [self[idx] for idx in range(start, stop, step)]
            return sample_slice
        # handle indices
        else:
            # If the index is out of range, raise IndexError
            if idx >= self.num_samples:
                raise IndexError(f"Index {idx} out of range")

            # support negative indices
            if idx < 0:
                idx += self.num_samples

                if idx < 0:
                    raise IndexError(f"Index {idx} out of range")

            # fetch the block sample index
            if self.use_digitize:
                block_idx = np.digitize(idx, self.block_bins)
            else:
                block_idx = idx // self.block_size
            sample_block = self.get_sample_block(block_idx)

            # use the local index to fetch the sample
            local_idx = idx - self.block_bins[block_idx]
            sample_idx = sample_block[local_idx]

            return sample_idx

    def __len__(self) -> int:
        return self.num_samples

    def __reduce__(self):
        """Add support for pickling. Needed due to functools.lru_cache."""
        # Return a tuple with a callable and arguments to recreate the object
        return (
            self.__class__,
            (
                self.dataset_size,
                self.num_samples,
                self.block_size,
                self.cache_maxsize,
                self.seed,
                self.shuffle,
                self.truncate_to_block_boundary,
            ),
        )

    def __reduce_ex__(self, protocol: int):
        # Optional method that defines the protocol version
        return self.__reduce__()

    def get_sample_block(self, block_idx: int) -> np.ndarray:
        """
        Returns a block of samples of size self.block_size, shuffled if needed.

        Args:
            block_idx (int): Index of the block to retrieve.

        Returns:
            np.ndarray: Array of sample indices.
        """
        if block_idx >= self.num_blocks:
            raise IndexError(
                f"block_idx {block_idx} is out of range. Maximum block_idx is {self.num_blocks-1}"
            )

        # recover index of original block (before shuffling)
        start_idx = self.block_idx_list[block_idx] * self.block_size
        end_idx = start_idx + self.block_size_list[block_idx]
        sample_block = np.arange(start_idx, end_idx)

        # shuffle if needed
        if self.shuffle:
            local_rng = np.random.RandomState(seed=self.seed + block_idx)
            sample_block = local_rng.permutation(sample_block)

        # project indices to the dataset size
        sample_block = sample_block % self.dataset_size

        return sample_block


class OnlineSampleMappingDataset(torch.utils.data.Dataset):
    """
    A dataset that uses OnlineSampleMapping to provide samples.
    """

    def __init__(
        self, dataset: torch.utils.data.Dataset, num_samples: int, **kwargs: Any
    ) -> None:
        self.dataset = dataset
        dataset_size = len(dataset)
        self.num_samples = num_samples
        self.mapping = OnlineSampleMapping(dataset_size=dataset_size, **kwargs)

    def __str__(self) -> str:
        return (
            f"OnlineSampleMappingDataset("
            f"dataset_size={len(self.dataset)}, "
            f"num_samples={self.num_samples}"
            f")"
        )

    def __getitem__(self, idx: int) -> Any:
        if isinstance(idx, slice):
            return [self.dataset[i] for i in self.mapping[idx]]
        else:
            return self.dataset[self.mapping[idx]]

    def __len__(self) -> int:
        return len(self.mapping)


class _InterleavedDatasetIndexer:
    """
    Indexer for interleaved datasets. Maps a global index to a dataset index and a local index within that dataset
    based on interleaved sampling.
    """

    def __init__(self, dataset_counts: List[int], num_samples: int) -> None:
        """
        Args:
            dataset_counts (List[int]): List of dataset sizes.
            num_samples (int): Total number of samples.
        """
        self.dataset_counts = dataset_counts
        self.cumsum = [0]
        for count in dataset_counts:
            self.cumsum.append(self.cumsum[-1] + count)
        self.total = self.cumsum[-1]
        self.num_samples = num_samples

    def map_index(self, idx: int) -> Tuple[int, int]:
        """
        Maps a global index to a dataset index and a local index within that dataset.

        Args:
            idx (int): Global index.

        Returns:
            Tuple[int, int]: Dataset index and local index.
        """
        cycle_idx = idx % self.total
        ds_idx = bisect.bisect_right(self.cumsum, cycle_idx) - 1
        local_offset = cycle_idx - self.cumsum[ds_idx]
        local_idx = (idx // self.total) * self.dataset_counts[ds_idx] + local_offset
        return ds_idx, local_idx

    def __len__(self) -> int:
        """
        Returns the total number of samples.

        Returns:
            int: Total number of samples.
        """
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[int, int]:
        """
        Returns the dataset index and local index for the given global index.

        Args:
            idx (int): Global index.

        Returns:
            Tuple[int, int]: Dataset index and local index.

        Raises:
            IndexError: If the index is out of range.
        """
        if isinstance(idx, slice):
            start = handle_index(self.num_samples, idx.start or 0)
            stop = handle_index(
                self.num_samples, idx.stop if idx.stop is not None else self.num_samples
            )
            step = idx.step or 1
            if stop > self.num_samples:
                raise IndexError(f"Index stop = {stop} out of range")
            return [self.map_index(i) for i in range(start, stop, step)]
        idx = handle_index(self.num_samples, idx)
        if idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range")
        return self.map_index(idx)

    def __str__(self) -> str:
        return (
            f"_InterleavedDatasetIndexer("
            f"dataset_counts={self.dataset_counts}, "
            f"total_samples={self.num_samples}, "
            f"total_size={self.total}"
            f")"
        )


class WeightedConcatOnlineDataset(torch.utils.data.Dataset):
    """
    A dataset that can concat and weight multiple datasets in an online manner.
    """

    def __init__(
        self,
        datasets: List[torch.utils.data.Dataset],
        weights: Optional[List[float]] = None,
        seed: int = 1,
        shuffle: bool = True,
        interleaved: bool = True,
        interleaved_block_size: int = 1000000,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            datasets (List[torch.utils.data.Dataset]): List of datasets to concatenate.
            weights (Optional[List[float]]): List of weights corresponding to each dataset. Defaults to None (which will be replaced with the relative size of each dataset).
            seed (int): Seed for random number generation. Defaults to 1.
            shuffle (bool): Whether to shuffle the samples. Defaults to True.
            interleaved (bool): Whether to interleave samples from datasets. Defaults to True.
            interleaved_block_size (int): Block size for interleaved sampling. Defaults to 1000000. IMPORTANT: If too small, the sampling frequency of each sample might be approximated (default value is likely to be good in most cases).
            **kwargs (Any): Additional arguments passed to `OnlineSampleMapping`.
        """
        # add to kwargs to pass to OnlineSampleMapping
        kwargs["seed"] = seed
        kwargs["shuffle"] = shuffle
        if len(datasets) == 0:
            raise ValueError("No datasets provided.")
        if weights is None:
            ds_len = [len(ds) for ds in datasets]
            ds_total_len = np.sum(ds_len)
            if ds_total_len == 0:
                raise ValueError("No samples in datasets.")
            weights = [ds_len[i] / ds_total_len for i in range(len(datasets))]
        # Normalize weights
        weights = np.array(weights, dtype=np.float32)
        if np.sum(weights) == 0.0:
            raise ValueError("Weights must be positive. " f"Got {weights}.")
        if np.sum(weights) != 1.0:
            logger.warning(f"Weights do not sum to 1.0. Normalizing weights: {weights}")
            weights /= np.sum(weights)
        if np.any(weights < 0):
            raise ValueError("Weights must be positive. " f"Got {weights}.")
        if np.any(weights == 0):
            logger.warning(
                "Weights contain zero values. This may lead to empty datasets."
            )

        num_samples = int(np.sum([len(d) for d in datasets]))

        self.datasets = datasets
        self.weights = weights = np.array(weights, dtype=np.float32)
        self.seed = seed
        self.shuffle = shuffle
        self.interleaved = interleaved
        self.interleaved_block_size = interleaved_block_size

        # check that the number of datasets and weights match
        if len(datasets) != len(weights):
            raise ValueError(
                "Number of datasets and weights must match. "
                f"Got {len(datasets)} datasets and {len(weights)} weights."
            )
        # check that the number of samples is positive
        if num_samples <= 0:
            raise ValueError(
                "Number of samples must be positive. " f"Got {num_samples} samples."
            )

        if interleaved_block_size < len(datasets):
            raise ValueError(
                "Interleaved block size must be larger than the number of datasets. "
                f"Got {interleaved_block_size} block size and {len(datasets)} datasets."
            )

            # sample datasets to match the number of samples
        weighted_dataset_num_samples = weights * num_samples
        weighted_dataset_num_samples = np.round(weighted_dataset_num_samples).astype(
            int
        )
        weighted_dataset_num_samples[-1] = num_samples - np.sum(
            weighted_dataset_num_samples[:-1]
        )
        self.weighted_datasets = [
            OnlineSampleMappingDataset(dataset, dataset_num_samples, **kwargs)
            for dataset, dataset_num_samples in zip(
                datasets, weighted_dataset_num_samples
            )
        ]
        # read back actual number of samples (to support truncation)
        weighted_dataset_num_samples = np.array(
            [len(d) for d in self.weighted_datasets]
        )
        # validate that the number of samples is correct
        if np.sum(weighted_dataset_num_samples) != num_samples:
            # will likely be reported when truncating to block boundary
            logger.warning(
                f"Number of samples in datasets {sum([len(d) for d in self.weighted_datasets])} does not match the number of samples {num_samples}"
            )
            num_samples = sum([len(d) for d in self.weighted_datasets])
        # recomute weights
        weights = weighted_dataset_num_samples / num_samples
        if not np.allclose(weights, self.weights, atol=1e-5):
            logger.warning(
                f"Requested weights {self.weights} do not match the actual weights {weights}"
            )

        self.weights = weights
        self.num_samples = num_samples

        # if interleaved_block_size is larger than num_samples, set it to num_samples or else some samples might be skipped when shuffle is True
        self.interleaved_block_size = interleaved_block_size = min(
            interleaved_block_size, num_samples
        )

        if interleaved:
            # support interleaved sampling
            # divide interleaved_block_size based on weights
            dataset_counts = [round(w * interleaved_block_size) for w in weights]
            dataset_counts[-1] = interleaved_block_size - np.sum(dataset_counts[:-1])
            self.interleaved_dataset_counts = dataset_counts
            self.interleaved_weights = np.array(dataset_counts) / sum(dataset_counts)
            if not np.allclose(weights, self.interleaved_weights, atol=1e-2):
                logger.warning(
                    f"Requested interleaved weights {self.interleaved_weights} do not match the actual weights {weights}"
                )
            dataset_indexer = _InterleavedDatasetIndexer(
                dataset_counts=dataset_counts,
                num_samples=num_samples,
            )
        else:
            dataset_indexer = _InterleavedDatasetIndexer(
                dataset_counts=weighted_dataset_num_samples,
                num_samples=num_samples,
            )
        self.dataset_indexer = dataset_indexer
        if shuffle:
            # shuffle the joint dataset indices
            self.dataset_sampler = OnlineSampleMappingDataset(
                dataset_indexer,
                num_samples=num_samples,
                block_size=interleaved_block_size,
                seed=seed,
                shuffle=shuffle,
                truncate_to_block_boundary=False,
            )
        else:
            # if not shuffling, just use the dataset indexer
            self.dataset_sampler = dataset_indexer

    def __str__(self) -> str:
        return (
            f"WeightedConcatOnlineDataset("
            f"num_datasets={len(self.datasets)}, "
            f"weights={self.weights}, "
            f"num_samples={self.num_samples}, "
            f"shuffle={self.shuffle}, "
            f"interleaved={self.interleaved}, "
            f"interleaved_block_size={self.interleaved_block_size}"
            f")"
        )

    def __getitem__(self, idx: int) -> Any:
        if isinstance(idx, slice):
            start, stop, step = (
                idx.start or 0,
                idx.stop or self.num_samples,
                idx.step or 1,
            )
            return [self[i] for i in range(start, stop, step)]

        # Map the index to a dataset using the weights
        # dataset_idx, sample_idx = self.get_dataset_sample_idx_from_idx(idx)
        ds_idx, sample_idx = self.dataset_sampler[idx]
        # dataset_indexer (and thus dataset_sampler) does not respect the dataset size, so we need to do it here
        sample_idx = sample_idx % len(self.weighted_datasets[ds_idx])
        return self.weighted_datasets[ds_idx][sample_idx]

    def __len__(self) -> int:
        return self.num_samples


class OffsetOnlineDataset(torch.utils.data.Dataset):
    """
    A dataset that can offset the samples from a given dataset, effectively reducing the dataset size, and starting from a given offset.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        offset: int = 0,
    ) -> None:
        self.dataset = dataset

        self.set_offset(offset)

    def __str__(self) -> str:
        return (
            f"OffsetOnlineDataset("
            f"dataset_size={len(self.dataset)}, "
            f"offset={self.offset}"
            f")"
        )

    def __getitem__(self, idx: int) -> Any:
        if isinstance(idx, slice):
            start, stop, step = (
                idx.start or 0,
                idx.stop or len(self.dataset),
                idx.step or 1,
            )
            return [self[i] for i in range(start, stop, step)]

        # add offset to the index
        sample_idx = handle_index(len(self), idx) + self.offset
        return self.dataset[sample_idx]

    def __len__(self) -> int:
        return len(self.dataset) - self.offset

    def set_offset(self, offset: int) -> "OffsetOnlineDataset":
        """
        Set the offset for the dataset.

        Args:
            offset (int): Offset to set.
        """
        if offset < 0:
            raise ValueError(f"Offset must be positive. Got {offset} offset.")
        if offset >= len(self.dataset):
            raise ValueError(
                f"Offset must be smaller than the dataset size. Got {offset} offset and {len(self.dataset)} dataset size."
            )
        self.offset = offset
        return self
