"""Example code based on https://github.com/huggingface/datasets/issues/7156.

TODO: use our own example data.
"""
import datasets
import torch.utils.data


def gen(shards):
    yield {"shards": shards}


def main():
    dataset = datasets.IterableDataset.from_generator(
        gen,
        gen_kwargs={'shards': list(range(25))}  # TODO: how to understand this?
    )
    # dataset = dataset.shuffle(buffer_size=1)  can individually shuffle here

    dataset = datasets.interleave_datasets(
        [dataset, dataset], probabilities=[0.5, 0.5], stopping_strategy="all_exhausted"
    )
    # dataset = dataset.shuffle(buffer_size=1)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        num_workers=8,
    )

    for i, batch in enumerate(dataloader):
        print(batch)
    print("New epoch")

    # dataset = dataset.set_epoch(1)
    dataset = dataset.shuffle(buffer_size=1)
    for i, batch in enumerate(dataloader):
        print(batch)



if __name__ == "__main__":
     main()