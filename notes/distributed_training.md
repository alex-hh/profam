### Map-style datasets

[Notes based on colab notebook here](https://colab.research.google.com/drive/1OFLZnX9y5QUFNONuvFsxOizq4M-tFvk-#scrollTo=Jflo2-6zAf0K)

Pytorch lightning will automatically apply distributed sampler to map-style datasets.
(Q. Will HF datasets also do something automatically?)


### Iterable datasets

#### split_dataset_by_node

[Notes based mainly on discussion thread here](https://github.com/huggingface/datasets/issues/6623)

The behaviour of the dataset after this function is applied depends on whether the
number of shards of the dataset is divisible by the world size:

if this is the case, then each epoch, n_shards // world_size shards will be assigned
to each device.

[Q. is it important in this case that each shard contains the same number of samples]

if n_shards is not divisible by world size, then each device will see all the shards,
and loop over the data in the same order. To ensure each device sees different data
the ith device takes only the ith sample.

This way each example is seen once and exactly once during distributed training,
though in terms of I/O, the dataset is effectively read / streamed twice.

In this case, if the total number of SAMPLES (i.e. datapoints) is not divisible by
the world size, then 'at least one of the nodes will get an empty / incomplete batch.
The data is not repeated in that case. If the training loop doesn't take this into
account it can lead to unexpected behaviours.'

[Note: to get an empty batch, I think total num samples % world size has to be less than
world size; I don't know if there is some way to handle this]

#### Edge-cases and solutions

If number of shards is not divisible by world size: do we need to handle overhanging samples?
Do we want number of shards to be divisible by world size? If so, we can repeat some
files up to a point where it is.

To avoid issues with iterable validation datasets, one solution would be to always use
map-style datasets for validation data.


### Questions

What exactly is the role of shards? In a non-distributed setting,
What's the easiest way to test this stuff? Perhaps something like the example
from the discussion thread?
Does interleave datasets sample without replacement? How exactly does e.g.
all exhausted work?

```python
import torch
import torch.distributed as dist
from datasets import Dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import DataLoader


# simulate a streaming dataset
num_shards = 1  # change here if you want to simulate a dataset made of many files/shards
ds = Dataset.from_dict({"x": [1, 2, 3, 4, 5]}).to_iterable_dataset(num_shards=num_shards)

# split the dataset for distributed training
dist.init_process_group()
rank, world_size = dist.get_rank(), dist.get_world_size()
ds = split_dataset_by_node(ds, rank=rank,world_size=world_size)
dl = DataLoader(ds)

exhausted = torch.zeros(world_size, dtype=torch.bool)

# IMPORTANT: Loop over the local dataset until the data from each rank has been exhausted
for x in cycle(chain(dl, ["end"])):
    if x == "end":
        exhausted[rank] = True
        continue
    # stop once the data from all the ranks are exhausted
    dist.all_reduce(exhausted)
    if torch.all(exhausted):
        break
    # do your forward pass + loss here
    # model.forward(...)
    print(x)
```
