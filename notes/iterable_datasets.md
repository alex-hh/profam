## Background: Iterators and iterables

An iterable (a class with an __iter__ method that specifies how to create a new iterator object). The simplest implementation of an __iter__ method would just return an iterator directly. Alternatively, __iter__ can be a generator function (a function containing yield statement). When called, a generator function returns a generator objects, which is a type of iterator

For loops in python loop over iterables by creating iterators (via iter(iterable)) and calling next on them.


## Iterable datasets in pytorch

The basic idea of iterable datasets is this: the dataset is an iterator

Each epoch we create a (new) iterator by calling iter(dataset)
This calls __iter__, returning a new iterator.
If HF datasets, the random generator associated with the iterator returned
by __iter__ is a function of the epoch (which is updated by calling set epoch),
as well as a basic seed, set by calling shuffle. This is safer than passing around
a global generator, since this generator could diverge between processes based on different
data being seen on different devices. However, this only matters if different devices
are iterating over the same data.

HF datasets also shuffles shards across devices, but this seems unnecessary.

N.B. some code removed for clarity:
```python
class DataLoader:
    def __iter__(self) -> '_BaseDataLoaderIter':
        if self.persistent_workers and self.num_workers > 0:
            if self._iterator is None:
                self._iterator = self._get_iterator()
            else:
                self._iterator._reset(self)
            return self._iterator
        else:
            # creates a new _SingleProcessDataLoaderIter and associated fetcher
            return self._get_iterator()


class _SingleProcessDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader):
        super().__init__(loader)
        assert self._timeout == 0
        assert self._num_workers == 0

        self._dataset_fetcher = _DatasetKind.create_fetcher(
            self._dataset_kind, self._dataset, self._auto_collation, self._collate_fn, self._drop_last)

    def _next_data(self):
        index = self._next_index()  # may raise StopIteration
        data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        if self._pin_memory:
            data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)
        return data

    def __next__(self) -> Any:
        with torch.autograd.profiler.record_function(self._profile_name):
            if self._sampler_iter is None:
                # TODO(https://github.com/pytorch/pytorch/issues/76750)
                self._reset()  # type: ignore[call-arg]
            data = self._next_data()

    def _reset(self, loader, first_iter=False):
        self._sampler_iter = iter(self._index_sampler)
        self._num_yielded = 0
        self._IterableDataset_len_called = loader._IterableDataset_len_called


class _IterableDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        super().__init__(dataset, auto_collation, collate_fn, drop_last)
        self.dataset_iter = iter(dataset)
        self.ended = False

    def fetch(self, possibly_batched_index):
        if self.ended:
            raise StopIteration
        # ... (fetching logic omitted for brevity)

```

If persistent_workers is True, the MultiProcessingDataLoaderIter will create a new
fetcher (thereby creating a new iterator).

## Iterable datasets in the Huggingface Datasets library

HF datasets constructs stateful iterable datasets. The main work is done in the
__iter__ function, which returns a new iterator.


An annotated, high-level version of an IterableDataset is:

```python

class ExamplesIterable(_BaseExamplesIterable):
    def __init__(self, generate_examples_fn: Callable[..., Tuple[Key, dict]], kwargs: dict):
        super().__init__()
        self.generate_examples_fn = generate_examples_fn
        self.kwargs = kwargs

    def _init_state_dict(self) -> dict:
        self._state_dict = {"shard_idx": 0, "shard_example_idx": 0}
        return self._state_dict

    def __iter__(self):
        shard_idx_start = self._state_dict["shard_idx"] if self._state_dict else 0
        for gen_kwags in islice(_split_gen_kwargs(self.kwargs, max_num_jobs=self.n_shards), shard_idx_start, None):
            shard_example_idx_start = self._state_dict["shard_example_idx"] if self._state_dict else 0
            for key_example in islice(self.generate_examples_fn(**gen_kwags), shard_example_idx_start, None):
                if self._state_dict:
                    self._state_dict["shard_example_idx"] += 1
                yield key_example
            if self._state_dict:
                self._state_dict["shard_idx"] += 1
                self._state_dict["shard_example_idx"] = 0

    def shuffle_data_sources(self, generator: np.random.Generator) -> "ExamplesIterable":
        return ShuffledDataSourcesExamplesIterable(self.generate_examples_fn, self.kwargs, generator)

    def shard_data_sources(self, worker_id: int, num_workers: int) -> "ExamplesIterable":
        """Keep only the requested shard."""
        gen_kwargs_list = _split_gen_kwargs(self.kwargs, max_num_jobs=self.n_shards)
        shard_indices = self.split_shard_indices_by_worker(worker_id, num_workers)
        requested_gen_kwargs = _merge_gen_kwargs([gen_kwargs_list[i] for i in shard_indices])
        return ExamplesIterable(self.generate_examples_fn, requested_gen_kwargs)

    @property
    def n_shards(self) -> int:
        return _number_of_shards_in_gen_kwargs(self.kwargs)

class ShuffledDataSourcesExamplesIterable(ExamplesIterable):
    def __init__(
        self, generate_examples_fn: Callable[..., Tuple[Key, dict]], kwargs: dict, generator: np.random.Generator
    ):
        super().__init__(generate_examples_fn, kwargs)
        self.generator = deepcopy(generator)

    def _init_state_dict(self) -> dict:
        self._state_dict = {"shard_idx": 0, "shard_example_idx": 0}
        return self._state_dict

    def __iter__(self):
        """Shuffle the kwargs order to shuffle shards"""
        rng = deepcopy(self.generator)
        kwargs_with_shuffled_shards = _shuffle_gen_kwargs(rng, self.kwargs)
        shard_idx_start = self._state_dict["shard_idx"] if self._state_dict else 0
        for gen_kwags in islice(
            _split_gen_kwargs(kwargs_with_shuffled_shards, max_num_jobs=self.n_shards), shard_idx_start, None
        ):
            shard_example_idx_start = self._state_dict["shard_example_idx"] if self._state_dict else 0
            for key_example in islice(self.generate_examples_fn(**gen_kwags), shard_example_idx_start, None):
                if self._state_dict:
                    self._state_dict["shard_example_idx"] += 1
                yield key_example
            if self._state_dict:
                self._state_dict["shard_idx"] += 1
                self._state_dict["shard_example_idx"] = 0

    def shard_data_sources(self, worker_id: int, num_workers: int) -> "ExamplesIterable":
        """Keep only the requested shard."""
        rng = deepcopy(self.generator)
        kwargs_with_shuffled_shards = _shuffle_gen_kwargs(rng, self.kwargs)
        return ExamplesIterable(self.generate_examples_fn, kwargs_with_shuffled_shards).shard_data_sources(
            worker_id, num_workers
        )

class IterableDataset(torch.utils.data.IterableDataset):

    def __init__(
        self,
        ex_iterable,
        formatting: Optional[FormattingConfig] = None,
        shuffling: Optional[ShufflingConfig] = None,
        distributed: Optional[DistributedConfig] = None,
    ):
        self._ex_iterable = copy.copy(ex_iterable)
        self._formatting = formatting
        self._shuffling = shuffling
        self._distributed = distributed
        self._starting_state_dict = None
        self._prepared_ex_iterable = self._prepare_ex_iterable_for_iteration()
        self._state_dict = self._ex_iterable._init_state_dict()

    def from_generator(
        generator: Callable,
        gen_kwargs: Optional[dict] = None,
    ):
        """
        >>> def gen(shards):
        ...     for shard in shards:
        ...         with open(shard) as f:
        ...             for line in f:
        ...                 yield {"line": line}
        ...
        >>> shards = [f"data{i}.txt" for i in range(32)]
        >>> ds = IterableDataset.from_generator(gen, gen_kwargs={"shards": shards})
        >>> ds = ds.shuffle(seed=42, buffer_size=10_000)  # shuffles the shards order + uses a shuffle buffer
        >>> from torch.utils.data import DataLoader
        >>> dataloader = DataLoader(ds.with_format("torch"), num_workers=4)  # give each worker a subset of 32/4=8 shards
        """
        # return GeneratorDatasetInputStream(gen_kwargs=gen_kwargs, streaming=True,
        # ).read()
        # basically equivalent to
        class Generator:
            generator: Callable
            def _generate_examples(self, **gen_kwargs):
                for idx, ex in enumerate(self.generator):
                    yield idx, ex

        # Q. where do shards come into play?
        ex_iterable = ExamplesIterable(Generator(generator)._generate_examples, gen_kwargs)
        return IterableDataset(ex_iterable)

    def _effective_generator(self):
        if self._shuffling and self.epoch == 0:
            return self._shuffling.generator
        elif self._shuffling:
            # Create effective seed using self.epoch (we subtract in order to avoir overflow in long_scalars)

            # sample a random number between 0 and 2^63-1, then subtract the epoch
            # the generator is the same each epoch, so the seeds are offset by 1
            effective_seed = deepcopy(self._shuffling.generator).integers(0, 1 << 63) - self.epoch
            # not shown: correct for negative seed
            return np.random.default_rng(effective_seed)
        else:
            raise ValueError("This dataset is not shuffled")

    @property
    def n_shards(self) -> int:
        if self._distributed and self._ex_iterable.n_shards % self._distributed.world_size == 0:
            return self._ex_iterable.n_shards // self._distributed.world_size
        return self._ex_iterable.n_shards

    def _prepare_ex_iterable_for_iteration(self, batch_size: int = 1, drop_last_batch: bool = False):
        ex_iterable = self._ex_iterable
        if self._shuffling:
            ex_iterable = ex_iterable.shuffle_data_sources(self._effective_generator())
        if self._distributed:
            rank = self._distributed.rank
            world_size = self._distributed.world_size
            if ex_iterable.n_shards % world_size == 0:
                ex_iterable = ex_iterable.shard_data_sources(rank, world_size)
            else:
                # not shown
        self._state_dict = ex_iterable._init_state_dict()
        if self._starting_state_dict:
            ex_iterable.load_state_dict(self._starting_state_dict)
        return ex_iterable

    def __iter__(self):
        ex_iterable = self._prepare_ex_iterable_for_iteration()
        if self._formatting:
            formatter = get_formatter(self._formatting.format_type, features=self.features)
            format_dict = cast_to_python_objects
        else:
            format_dict = None
        for key, example in ex_iterable:
            yield format_dict(example) if format_dict else example

    def shuffle(
        self, seed=None, generator: Optional[np.random.Generator] = None
    ):
        # Note that this doesn't call shuffle_data_sources
        if generator is None:
            generator = np.random.default_rng(seed)
        shuffling = ShufflingConfig(generator=generator, _original_seed=seed)
        return Iterable
```



### Transforming Iterable Datasets

There are two types of transformations that can be performed on an iterable dataset:

* state-changing transformations change the state, which affects the way in which an iterator is constructed from the underlying iterable
* iterable-changing transformations act directly on the underlying iterable.

shuffle, split_dataset_by_node are examples of state-changing transformations

take is an example of an iterable-changing transformation.

Composition of transformations has to propagate the desired behaviours. For example, consider shuffling then performing take. Shuffle just sets shuffling config, which will be used by shuffle_data_sources on the example iterable. The simplest form of shuffling just shuffles the shards. So if we call shuffle after take
