from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from transformers import MistralForCausalLM, MistralConfig
from torchmetrics import MeanMetric

class MistralLitModule(LightningModule):
    def __init__(self, config: MistralConfig, compile: bool = False) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.model = MistralForCausalLM(config)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)


    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self(batch["input_ids"], batch["attention_mask"], batch["input_ids"])
        loss = outputs.loss
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor],  batch_idx: int) -> torch.Tensor:
        outputs = self(batch["input_ids"], batch["attention_mask"], labels=batch["input_ids"])
        loss = outputs.loss
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch: Dict[str, torch.Tensor],  batch_idx: int) -> torch.Tensor:
        outputs = self(batch["input_ids"], batch["attention_mask"], labels=batch["input_ids"])
        loss = outputs.loss
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-5, weight_decay=0.01)
        return {"optimizer": optimizer}