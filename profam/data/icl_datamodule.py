"""Lightning data module for the supervised ICL fine-tune.

Deliberately minimal: one ICL document per sample, no multi-dataset packing.
Train / val datasets are :class:`profam.data.builders.proteingym_icl.ProteinGymICLDataset`
instances over the train and held-out cluster splits respectively.
"""

from __future__ import annotations

from typing import Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from profam.data.collators import ICLDocumentBatchCollator
from profam.data.tokenizers import ProFamTokenizer


class ICLDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset],
        tokenizer: ProFamTokenizer,
        batch_size: int = 1,
        num_workers: int = 0,
        ignore_gaps: bool = True,
        prefetch_factor: Optional[int] = None,
        test_dataset: Optional[Dataset] = None,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.prefetch_factor = prefetch_factor if num_workers else None
        self.collator = ICLDocumentBatchCollator(
            tokenizer=tokenizer,
            ignore_gaps=ignore_gaps,
        )
        # Attach tokenizer to ICL datasets that don't have one yet.
        for ds in (train_dataset, val_dataset, test_dataset):
            if ds is None:
                continue
            if getattr(ds, "_tokenizer", None) is None and hasattr(ds, "_tokenizer"):
                ds._tokenizer = tokenizer

    def setup(self, stage: Optional[str] = None) -> None:
        return None

    def _make_loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            collate_fn=self.collator,
            persistent_workers=self.num_workers > 1,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset is None:
            return None
        return self._make_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_dataset is None:
            return None
        return self._make_loader(self.test_dataset, shuffle=False)
