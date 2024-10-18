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
