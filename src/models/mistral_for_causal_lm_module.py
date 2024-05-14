import torch
from pytorch_lightning import LightningModule
from transformers import MistralConfig, MistralForCausalLM

class ProteinModel(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = MistralForCausalLM(config)
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask=attention_mask)

    def training_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], batch["attention_mask"])
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
        return optimizer