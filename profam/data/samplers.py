import random
from typing import Any, Callable, List

from torch.utils.data import BatchSampler, Dataset


class MaxTokensDynamicBatchSampler(BatchSampler):
    """
    Splits the dataset into dynamic batches based on token lengths so that each
    batch contains approximately max_tokens tokens. The batches are then sharded
    across distributed processes using round-robin assignment.
    """

    def __init__(
        self,
        dataset: Dataset,
        size_fn: Callable[[Any], int],  # added type hint for size_fn
        world_size: int,
        rank: int,
        max_tokens: int = None,
        batch_size: int = None,
    ):
        """
        Args:
            dataset (Dataset): The dataset to sample from.
            size_fn (Callable[[Any], int]): Function to compute the size (number of tokens)
                of a given sample.
            world_size (int): Total number of distributed processes.
            rank (int): Rank of the current process.
            max_tokens (int, optional): Maximum number of tokens allowed per batch.
            batch_size (int, optional): Number of samples per batch.

        Note:
            Exactly one of max_tokens or batch_size must be specified.
        """
        self.dataset = dataset
        self.size_fn = size_fn  # now using the provided size_fn function
        self.world_size = world_size
        self.rank = rank
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        if self.max_tokens is None and self.batch_size is None:
            raise ValueError("Either max_tokens or batch_size must be specified.")
        if self.max_tokens is not None and self.batch_size is not None:
            raise ValueError(
                "Only one of max_tokens or batch_size should be specified."
            )

    def __iter__(self):
        """
        Yields:
            List[int]: Next batch of sample indices assigned to this process.
        """
        batch = []
        if self.max_tokens is not None:
            # If max_tokens is specified, yield batches based on token counts
            current_tokens = 0
            for idx in range(self.rank, len(self.dataset), self.world_size):
                tokens = self.size_fn(self.dataset[idx])  # compute tokens on-the-fly
                # Start a new batch if adding the current sample exceeds max_tokens
                if batch and (current_tokens + tokens > self.max_tokens):
                    yield batch
                    batch = [idx]
                    current_tokens = tokens
                else:
                    batch.append(idx)
                    current_tokens += tokens
            # Yield the last batch if it exists
            if batch:
                yield batch
        else:
            # If batch_size is specified, yield batches of that size
            for idx in range(self.rank, len(self.dataset), self.world_size):
                batch.append(idx)
                if len(batch) >= self.batch_size:
                    yield batch
                    batch = []
                if batch:
                    yield batch

    def __len__(self):
        """
        Dynamic batch sampler does not know how many batches will be yielded for this process.
        """
        return None
