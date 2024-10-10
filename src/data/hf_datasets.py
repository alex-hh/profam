import copy
from typing import List, Optional

import numpy as np
from datasets.info import DatasetInfo
from datasets.iterable_dataset import (
    CyclingMultiSourcesExamplesIterable,
    IterableDataset,
    RandomlyCyclingMultiSourcesExamplesIterable,
    _BaseExamplesIterable,
)
from datasets.splits import NamedSplit


class RepeatExamplesIterable(_BaseExamplesIterable):
    """
    Iterable that repeats the underlying iterable a given number of times.
    """

    def __init__(
        self,
        ex_iterable: _BaseExamplesIterable,
        num_times: int,
    ):
        super().__init__()
        self.ex_iterable = ex_iterable
        self.num_times = num_times

    def _init_state_dict(self) -> dict:
        self._state_dict = {
            "repeat_index": 0,
            "ex_iterable": self.ex_iterable._init_state_dict(),
        }
        return self._state_dict

    def __iter__(self):
        repeat_index = self._state_dict["repeat_index"] if self._state_dict else 0
        while True:
            if self.num_times and repeat_index >= max(self.num_times, 0):
                break
            yield from self.ex_iterable
            repeat_index += 1
            if self._state_dict:
                self._state_dict["repeat_index"] = repeat_index
                self._state_dict["ex_iterable"] = self.ex_iterable._init_state_dict()

    def shuffle_data_sources(
        self, generator: np.random.Generator
    ) -> "RepeatExamplesIterable":
        """Shuffle the underlying iterable, then repeat."""
        return RepeatExamplesIterable(
            self.ex_iterable.shuffle_data_sources(generator), num_times=self.num_times
        )

    def shard_data_sources(
        self, worker_id: int, num_workers: int
    ) -> "RepeatExamplesIterable":
        """Shard, then repeat shards."""
        return RepeatExamplesIterable(
            self.ex_iterable.shard_data_sources(worker_id, num_workers),
            num_times=self.num_times,
        )

    @property
    def n_shards(self) -> int:
        return self.ex_iterable.n_shards


def repeat(
    dataset: IterableDataset, num_times: Optional[int] = None
) -> IterableDataset:
    return IterableDataset(
        ex_iterable=RepeatExamplesIterable(dataset._ex_iterable, num_times=num_times),
        info=dataset._info,
        split=dataset._split,
        formatting=dataset._formatting,
        shuffling=copy.deepcopy(dataset._shuffling),
        distributed=copy.deepcopy(dataset._distributed),
        token_per_repo_id=dataset._token_per_repo_id,
    )


def recklessly_interleave_datasets(
    datasets: List[IterableDataset],
    probabilities: Optional[List[float]] = None,
    seed: Optional[int] = None,
    split: Optional[NamedSplit] = None,
    stopping_strategy: str = "all_exhausted",
):
    """interleave_datasets without any feature checks. Be careful."""

    ex_iterables = [copy.deepcopy(d._ex_iterable) for d in datasets]
    # Use cycling or random cycling of sources
    if probabilities is None:
        ex_iterable = CyclingMultiSourcesExamplesIterable(
            ex_iterables, stopping_strategy=stopping_strategy
        )
    else:
        generator = np.random.default_rng(seed)
        ex_iterable = RandomlyCyclingMultiSourcesExamplesIterable(
            ex_iterables,
            generator=generator,
            probabilities=probabilities,
            stopping_strategy=stopping_strategy,
        )
    # Set new info - we update the features
    # setting the features also ensures to fill missing columns with None
    if info is None:
        info = DatasetInfo.from_merge([d.info for d in datasets])
    else:
        info = info.copy()
    # Get all the auth tokens per repository - in case the datasets come from different private repositories
    token_per_repo_id = {
        repo_id: token
        for dataset in datasets
        for repo_id, token in dataset._token_per_repo_id.items()
    }
    # Return new daset
    return IterableDataset(
        ex_iterable=ex_iterable,
        info=info,
        split=split,
        token_per_repo_id=token_per_repo_id,
    )


class InterleavedIterableDataset:
    def __init__(
        self,
        datasets: List[IterableDataset],
        probs: List[float],
        stopping_condition: str = "all_exhausted",
    ):
        self.datasets = datasets
        self.probs = probs
        self.stopping_condition = stopping_condition

    def __iter__(self):
        return interleave_datasets(self.datasets, self.probs, self.stopping_condition)


def interleave_datasets(
    datasets: List[IterableDataset],
    probs: List[float],
    stopping_condition: str = "all_exhausted",
):
    """Simple, understandable implementation of interleave_datasets.

    We don't do any feature checks as they can add headaches.

    Handling like this ensures we don't mess with the formatting of the
    constituent datasets.
    One complexifying factor is that we need to call split_datasets_by_node
    on individual datasets before interleaving them - but this is in a sense
    more explicit about what is going on.
    """
    assert sum(probs) == 1
    assert stopping_condition in ["all_exhausted", "first_exhausted"]
    dataset_iterators = [iter(ds) for ds in datasets]
    is_exhausted = [False] * len(datasets)
    while not all(is_exhausted):
        dataset_index = np.random.choice(len(datasets), p=probs)
        try:
            yield next(dataset_iterators[dataset_index])
        except StopIteration:
            is_exhausted[dataset_index] = True
            if stopping_condition == "first_exhausted":
                break
            else:
                # TODO: add feature request for reset method on iterators
                reset_dataset_iterator(dataset_iterators[dataset_index])
