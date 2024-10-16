"""Example code based on https://github.com/huggingface/datasets/issues/7156.

The issue actually points out a particular error: that a dataset
passed to interleave_datasets is not shuffled even if shuffle was
called on that dataset before interleaving.

The problem is resolved by calling shuffle after interleaving.ß
"""
import datasets
import torch.utils.data


def gen(shards):
    yield {"shards": shards}


def main():
    dataset = datasets.IterableDataset.from_generator(
        gen, gen_kwargs={"shards": list(range(25))}  # TODO: how to understand this?
    )
    # THIS IS THE LINE THAT IS IGNORED ATM
    # dataset = dataset.shuffle(buffer_size=1)

    dataset = datasets.interleave_datasets(
        [dataset, dataset], probabilities=[0.5, 0.5], stopping_strategy="all_exhausted"
    )
    print(dataset.n_shards)
    dataset = dataset.shuffle(buffer_size=1)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        num_workers=0,
    )

    for i, batch in enumerate(dataloader):
        print(batch)
    print("\nNew epoch")

    dataset = dataset.set_epoch(1)

    for i, batch in enumerate(dataloader):
        print(batch)


if __name__ == "__main__":
    main()
