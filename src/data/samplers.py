import random
from torch.utils.data import BatchSampler
from typing import List

class MaxTokensDynamicBatchSampler(BatchSampler):
    """
    Splits the dataset into dynamic batches based on token lengths so that each
    batch contains approximately max_tokens tokens. The batches are then sharded
    across distributed processes using round-robin assignment.
    """
    def __init__(self, token_lens: List[int], max_tokens: int, world_size: int, rank: int, shuffle: bool = False, seed: int = 0, drop_last: bool = False):
        """
        Args:
            token_lens (List[int]): List containing token lengths for each sample.
            max_tokens (int): Maximum number of tokens allowed per batch.
            world_size (int): Total number of distributed processes.
            rank (int): Rank of the current process.
            shuffle (bool, optional): Whether to shuffle the data before batching. Defaults to False.
            seed (int, optional): Random seed used for shuffling. Defaults to 0.
            drop_last (bool, optional): If True, drop the final partial batch if it doesn't meet criteria. Defaults to False.
        """
        # TODO: this is not efficient for massive datasets, as it computes batches. An online version would be better.
        self.token_lens = token_lens
        self.max_tokens = max_tokens
        self.world_size = world_size
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.num_samples = len(token_lens)
        
        # compute the batches based on token lengths
        batches = self._compute_batches(list(range(self.num_samples)))
        # make sure all ranks see the same number of batches if needed
        if self.drop_last:
            N = len(batches) 
            batches = batches[:N - (N % self.world_size)]
        # filter batches for the current rank
        self.batches = batches[self.rank::self.world_size]

    def _compute_batches(self, indices):
        """
        Computes batches based on the cumulative token lengths, ensuring that
        the total tokens in each batch do not exceed max_tokens.

        Args:
            indices (List[int]): List of sample indices to compute batches from.
        
        Returns:
            List[List[int]]: A list of batches, where each batch is a list of sample indices.
        """
        batches = []
        current_batch = []
        current_tokens = 0
        for idx in indices:
            token_count = self.token_lens[idx]
            if current_batch and (current_tokens + token_count > self.max_tokens):
                batches.append(current_batch)
                current_batch = [idx]
                current_tokens = token_count
            else:
                current_batch.append(idx)
                current_tokens += token_count
        if current_batch:
            if not self.drop_last or (current_tokens >= self.max_tokens):
                batches.append(current_batch)
        return batches

    def __iter__(self):
        """
        Yields:
            List[int]: Next batch of sample indices assigned to this process.
        """
        for batch in self.batches:
            yield batch

    def __len__(self):
        """
        Returns:
            int: Number of batches that will be yielded for this process.
        """
        return len(self.batches)
