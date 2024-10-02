Main question:

In distributed setups we can end up with different devices seeing different
numbers of samples. Is this necessarily an issue?

Maybe the toy test example covers this case exactly..


### Map-style datasets

[Notes based on colab notebook here](https://colab.research.google.com/drive/1OFLZnX9y5QUFNONuvFsxOizq4M-tFvk-#scrollTo=Jflo2-6zAf0K)

Pytorch lightning trainer will automatically apply distributed sampler to map-style datasets.
(Q. Will HF datasets also do something automatically?)

Distributed Sampler first checks if the dataset size is divisible by the number of devices,
adds extra (repeated) samples in case it is not, then divides the samples evenly between
the devices. [source](https://discuss.pytorch.org/t/distributedsampler/90205)

This leads to possible slight environment dependence of evaluation metrics:
exactly which data is evaluated on depends on the number of devices. One way around
this would be to choose a fixed number of samples that is divisible by all the world
sizes you want to work with (e.g. is divisible by 32, for example, and therefore compatible
with world size 1,2,4,8,16)

More info: https://github.com/pytorch/pytorch/issues/22584
Possible workaround: https://github.com/SeungjunNah/DeepDeblur-PyTorch/blob/master/src/data/sampler.py

### Iterable datasets

If you don't do anything, you'll get duplicated data. The standard solution is to shuffle the
data in the same way on each device, then only train on the ith datapoint on the ith device.
This is implemented automatically by the split_dataset_by_node function in datasets, although
whether it does exactly this depends on the way the dataset is sharded.

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

[Question: in what settings exactly can this lead to timeouts?]

[Note: to get an empty batch, I think total num samples % world size has to be less than
world size; I don't know if there is some way to handle this case]

#### Undesirable situations

If device 1 is waiting for a gradient update from device 0 that's never going to happen
because device 0 has finished training, potentially that causes a hang (don't understand
if this is what is described exactly:)

https://github.com/pytorch/pytorch/issues/22584

#### Possible solutions

A relatively simple solution for training:
ids.repeat(None).take(n_samples_per_epoch)
However repeat isn't yet implemented
https://github.com/huggingface/datasets/issues/7192

If we want to ensure that the number of shards is divisible by world size, we can repeat some
files up to a point where it is.

To avoid issues with iterable validation datasets, one solution would be to always use
map-style datasets for validation data.

'My recommendation for these situations where you don't know the total number of samples
apriori is to, configure the iterable dataset to yield a fixed number of samples before
raising StopIteration, and if necessary, repeat/reshuffle samples to hit that number'

One solution for training would be to calculate the total number of samples across
all datasets using the appropriate index files, then determine a number of samples
that each device should run to avoid any overhang when taking the ith sample on each
device. This would need index files to be maintained accurately.

This isn't robust because of filtering

Similar-ish solution for val is to just use map datasets and slice the dataset so
that the number of samples is divisible by 32. If we have fewer samples than this,
maybe we evenly repeat the samples so that the number of samples is divisible by 32,
or we just run on a single device?

We could set up an infinite loop with interleave datasets by adding an extra dataset
with probability 0, then set a fixed number of steps.
https://github.com/huggingface/datasets/issues/7147

one can already set the max number of iterations per dataset by doing dataset.take(n) on the dataset that should only have n samples.

so we want to calculate total number of samples in interleaved dataset - but this
might also be very challenging...


### Questions

* What exactly is the role of shards? In a non-distributed setting, do batches form
across shards if the number of samples in a shard is not divisible by batch size?
* Can we access the shards associated with each dataset? Based on that can we
  calculate the total number of samples?
* What's the easiest way to test this stuff? Perhaps something like the example
from the discussion thread?
* Does interleave datasets sample without replacement? How exactly does e.g.
all exhausted work?
    https://github.com/huggingface/datasets/pull/4831
* If number of shards is divisible by world size, is there any way to force the
other behaviour of split_dataset_by_node (i.e. each device seeing all shards).
* If number of shards is divisible by world size, do we need each shard to contain
the same number of samples?
    https://github.com/huggingface/datasets/issues/6437
    'If the shards don't contain the same number of examples, then some workers [devices] might end up with more examples than others.'
* If number of shards is not divisible by world size: do we need to handle overhanging samples?
Is the only real issue when one device gets an empty batch? Or are incomplete batches also problematic?

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

### Relevant issues

Datasets

https://github.com/huggingface/datasets/issues/6594

Changing from stepping shards to stepping samples means that every single process reads ALL
of the shards. This was never an intended default for sharded training, shards gain their performance
advantage in large scale distributed training by explicitly avoiding the need to have every process
overlapping in the data they read, by default, only the data allocated to each process via their
assigned shards should be read in each pass of the dataset.

Using a large scale CLIP example, some of the larger datasets have 10-20k shards across 100+TB of data.
Training with 1000 GPUs we are switching between reading 100 terabytes per epoch to 100 petabytes
if say change 20k % 1000 and drop one gpu-node to 20k % 992.

The 'step over samples' case might be worth the overhead in specific validation scenarios where
gaurantees of at least/most once samples seen are more important and do not make up a significant
portion of train time or are done in smaller world sizes outside of train.

https://github.com/huggingface/datasets/issues/6437
https://github.com/huggingface/datasets/issues/6623#issuecomment-2379458138
https://github.com/huggingface/datasets/issues/6719
https://github.com/huggingface/datasets/issues/7147