### Formatting

When using HF datasets, it's very important not to convert from arrays to lists and back (or to tensors);
this can be very slow.

For our own datasets, we can use return_tensors=True in preprocess_protein_data.

https://github.com/huggingface/datasets/issues/625#issuecomment-696595524
Casting from arrow to numpy can be 100x faster than casting from arrow to list.

Not completely straightforward in the non-iterable case:
Rlevant issues (on-the-fly transformation is difficult with non iterable datasets and formatting)

https://github.com/huggingface/datasets/issues/5841
https://github.com/huggingface/datasets/issues/6833
https://github.com/huggingface/datasets/issues/5910
https://github.com/huggingface/datasets/issues/6012

Numpy appears to be the fastest format to use:
https://github.com/huggingface/datasets/issues/5841#issuecomment-1547644879

We can assume that conversion of numpy tensors to pytorch
is handled fairly optimally by the collator.


### Feature type and serialisation

I'm unsure whether this is important for performance.

However, presumably if I could directly load an array of backbone coords
rather than having to concatenate it, that would surely have some performance
benefit.

Relevance of feature types and formats:
https://github.com/huggingface/datasets/issues/625#issuecomment-834179496

https://github.com/huggingface/datasets/pull/2922

If your data type contains a list of objects, then you want to use the Sequence feature.

On array2d / array3d: https://github.com/huggingface/datasets/issues/5841
