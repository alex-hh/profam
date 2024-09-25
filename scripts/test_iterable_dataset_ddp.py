"""Example code based on https://github.com/huggingface/datasets/issues/7156.

The issue actually points out a particular error: that a dataset
passed to interleave_datasets is not shuffled even if shuffle was
called on that dataset before interleaving.

The problem is resolved by calling shuffle after interleaving.

from datasets.distributed import split_dataset_by_node
ids = ds.to_iterable_dataset(num_shards=512)
ids = ids.shuffle(buffer_size=10_000)  # will shuffle the shards order and use a shuffle buffer when you start iterating
ids = split_dataset_by_node(ds, world_size=8, rank=0)  # will keep only 512 / 8 = 64 shards from the shuffled lists of shards when you start iterating
dataloader = torch.utils.data.DataLoader(ids, num_workers=4)  # will assign 64 / 4 = 16 shards from this node's list of shards to each worker when you start iterating
for example in ids:
    pass

def _effective_generator(self):
    if self._shuffling and self.epoch == 0:
        return self._shuffling.generator
    elif self._shuffling:
        # Create effective seed using self.epoch (we subtract in order to avoir overflow in long_scalars)
        effective_seed = deepcopy(self._shuffling.generator).integers(0, 1 << 63) - self.epoch
        effective_seed = (1 << 63) + effective_seed if effective_seed < 0 else effective_seed
        return np.random.default_rng(effective_seed)
    else:
        raise ValueError("This dataset is not shuffled")

def _prepare_ex_iterable_for_iteration(
    self, batch_size: int = 1, drop_last_batch: bool = False
) -> _BaseExamplesIterable:
    ex_iterable = self._ex_iterable
    if self._formatting and (ex_iterable.iter_arrow or self._formatting.format_type == "arrow"):
        ex_iterable = RebatchedArrowExamplesIterable(
            ex_iterable, batch_size=batch_size, drop_last_batch=drop_last_batch
        )
    if self._shuffling:
        ex_iterable = ex_iterable.shuffle_data_sources(self._effective_generator())
    else:
        ex_iterable = ex_iterable

    if self._distributed:
        rank = self._distributed.rank
        world_size = self._distributed.world_size
        if ex_iterable.n_shards % world_size == 0:
            if self._is_main_process():
                n_shards_per_node = ex_iterable.n_shards // world_size
                plural = "s" if n_shards_per_node > 1 else ""
                logger.info(
                    f"Assigning {n_shards_per_node} shard{plural} (or data source{plural}) of the dataset to each node."
                )
            ex_iterable = ex_iterable.shard_data_sources(rank, world_size)
        else:
            if self._is_main_process():
                logger.info(
                    f"Assigning 1 out of {world_size} examples of the dataset to each node. The others are skipped during the iteration."
                )
                logger.info(
                    f"It is more optimized to distribute the dataset shards (or data sources) across nodes. "
                    f"You can do that by using a dataset with number of shards that is a factor of world_size={world_size}. "
                    f"The current dataset has {ex_iterable.n_shards} which is not a factor of {world_size}"
                )
            ex_iterable = StepExamplesIterable(ex_iterable, step=world_size, offset=rank)

    self._state_dict = ex_iterable._init_state_dict()
    if self._starting_state_dict:
        ex_iterable.load_state_dict(self._starting_state_dict)
    return ex_iterable
"""
import datasets
import torch.utils.data
from datasets.distributed import split_dataset_by_node


def gen(shards):
    yield {"shards": shards}


def main():
    dataset1 = datasets.IterableDataset.from_generator(
        gen, gen_kwargs={"shards": list(range(24))}  # TODO: how to understand this?
    )
    dataset2 = datasets.IterableDataset.from_generator(
        gen, gen_kwargs={"shards": list(range(100,124))}  # TODO: how to understand this?
    )
    print(dataset1.n_shards)
    # THIS IS THE LINE THAT IS IGNORED ATM
    # dataset = dataset.shuffle(buffer_size=1)

    # THE BIT IM NOT SURE ABOUT IS HOW INTERLEAVE DATASETS INTERACTS WITH SHARDING/SPLIT BY NODE
    # DIsabling probabilities makes a bit simpler
    # probabilities=[0.5, 0.5]
    dataset = datasets.interleave_datasets(
        [dataset1, dataset2], stopping_strategy="all_exhausted"
    )
    print(dataset.n_shards)
    dataset = dataset.shuffle(buffer_size=1, seed=42)

    ds1 = split_dataset_by_node(dataset, world_size=2, rank=0)
    ds2 = split_dataset_by_node(dataset, world_size=2, rank=1)

    dataloader1 = torch.utils.data.DataLoader(
        ds1,
        batch_size=4,
        num_workers=0,
    )
    dataloader2 = torch.utils.data.DataLoader(
        ds2,
        batch_size=4,
        num_workers=0,
    )

    print("Dataloader 1")
    for i, batch in enumerate(dataloader1):
        print(batch)

    print("Dataloader 2")
    for i, batch in enumerate(dataloader2):
        print(batch)
    print("\nNew epoch")

    dataset = dataset.set_epoch(1)
    # TODO: understand what set_epoch does - because it seems to jumble shards...
    # will keep only 512 / 8 = 64 shards from the shuffled lists of shards when you start iterating
    ds1 = ds1.set_epoch(1)
    ds2 = ds2.set_epoch(1)

    epoch_2_shards = []
    print("Dataloader 1")
    for i, batch in enumerate(dataloader1):
        print(batch)
        epoch_2_shards.append(batch["shards"][0])

    print("Dataloader 2")
    for i, batch in enumerate(dataloader2):
        print(batch)
        epoch_2_shards.append(batch["shards"][0])
    
    # print(torch.cat(epoch_2_shards).sort().values)
    assert (torch.cat(epoch_2_shards).sort().values == torch.cat([torch.arange(24), torch.arange(100,124)])).all()

if __name__ == "__main__":
    main()
