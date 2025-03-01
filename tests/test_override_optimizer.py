import os
import tempfile

import pytest
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset
from transformers import LlamaConfig, PreTrainedTokenizerFast

from src.models.llama import LlamaLitModule


@pytest.fixture
def llama_config():
    """Create a small LlamaConfig for testing."""
    return LlamaConfig(
        vocab_size=1000,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=128,
    )


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=None,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )
    tokenizer.vocab_size = 1000
    return tokenizer


@pytest.fixture
def tiny_dataset():
    """Create a tiny dataset for testing."""
    # Create a small dataset with 10 samples
    batch_size = 2
    seq_length = 20
    
    # Create random input_ids, attention_mask, and labels
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones_like(input_ids)
    labels = torch.randint(0, 1000, (batch_size, seq_length))
    
    # Create a dataset
    dataset = TensorDataset(input_ids, attention_mask, labels)
    return dataset


class TinyDataModule:
    """A tiny data module for testing."""
    
    def __init__(self, dataset):
        self.dataset = dataset
    
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=2)
    
    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=2)


def test_override_optimizer_with_lightning(llama_config, mock_tokenizer, tiny_dataset):
    """Test override_optimizer_on_load with actual Lightning training and checkpoint loading."""
    # Create a temporary directory for checkpoints
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initial training parameters
        initial_lr = 3e-4
        initial_scheduler = "cosine"
        
        # Create the model with initial parameters
        model = LlamaLitModule(
            config=llama_config,
            tokenizer=mock_tokenizer,
            lr=initial_lr,
            scheduler_name=initial_scheduler,
            num_warmup_steps=10,
            num_training_steps=100,
        )
        
        # Override the forward method to handle our simple dataset
        def custom_forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
            # Simple language modeling loss
            logits = torch.randn(input_ids.shape[0], input_ids.shape[1], llama_config.vocab_size, device=input_ids.device)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, llama_config.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )
            return type('obj', (object,), {'loss': loss})
        
        # Patch the forward method
        model.forward = custom_forward.__get__(model)
        
        # Create a data module
        data_module = TinyDataModule(tiny_dataset)
        
        # Create a checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=tmpdir,
            filename="model",
            save_top_k=1,
            save_last=True,
        )
        
        # Create a trainer
        trainer = Trainer(
            max_steps=1,
            callbacks=[checkpoint_callback],
            enable_checkpointing=True,
            logger=False,
            enable_progress_bar=False,
        )
        
        # Train the model for one step
        trainer.fit(model, data_module)
        
        # Get the checkpoint path
        checkpoint_path = os.path.join(tmpdir, "model.ckpt")
        
        # Verify the checkpoint exists
        assert os.path.exists(checkpoint_path)
        
        # Load the checkpoint to verify it contains optimizer and scheduler states
        checkpoint = torch.load(checkpoint_path)
        assert "optimizer_states" in checkpoint
        assert "lr_schedulers" in checkpoint
        
        # Verify the initial learning rate in the checkpoint
        assert checkpoint["optimizer_states"][0]["param_groups"][0]["lr"] == initial_lr
        
        # New training parameters
        new_lr = 1e-5
        new_scheduler = "linear"
        
        # Case 1: Resume with override_optimizer_on_load=False (default)
        model_without_override = LlamaLitModule(
            config=llama_config,
            tokenizer=mock_tokenizer,
            lr=new_lr,
            scheduler_name=new_scheduler,
            num_warmup_steps=5,
            num_training_steps=50,
            override_optimizer_on_load=False,
        )
        
        # Patch the forward method
        model_without_override.forward = custom_forward.__get__(model_without_override)
        
        # Create a new trainer
        trainer_without_override = Trainer(
            max_steps=2,  # One more step than before
            enable_checkpointing=True,
            logger=False,
            enable_progress_bar=False,
        )
        
        # Resume training without override
        trainer_without_override.fit(model_without_override, data_module, ckpt_path=checkpoint_path)
        
        # Get the optimizer from the trainer
        optimizer_without_override = trainer_without_override.optimizers[0]
        
        # The learning rate should still be the initial one (from the checkpoint)
        assert optimizer_without_override.param_groups[0]["lr"] == initial_lr
        
        # Case 2: Resume with override_optimizer_on_load=True
        model_with_override = LlamaLitModule(
            config=llama_config,
            tokenizer=mock_tokenizer,
            lr=new_lr,
            scheduler_name=new_scheduler,
            num_warmup_steps=5,
            num_training_steps=50,
            override_optimizer_on_load=True,
        )
        
        # Patch the forward method
        model_with_override.forward = custom_forward.__get__(model_with_override)
        
        # Create a new trainer
        trainer_with_override = Trainer(
            max_steps=2,  # One more step than before
            enable_checkpointing=True,
            logger=False,
            enable_progress_bar=False,
        )
        
        # Resume training with override
        trainer_with_override.fit(model_with_override, data_module, ckpt_path=checkpoint_path)
        
        # Get the optimizer from the trainer
        optimizer_with_override = trainer_with_override.optimizers[0]
        
        # The learning rate should be the new one (override worked)
        assert optimizer_with_override.param_groups[0]["lr"] == new_lr


def test_override_optimizer_on_load(llama_config, mock_tokenizer):
    """Test that override_optimizer_on_load works correctly."""
    # Create a model with initial parameters
    initial_lr = 3e-4
    initial_model = LlamaLitModule(
        config=llama_config,
        tokenizer=mock_tokenizer,
        lr=initial_lr,
        scheduler_name="cosine",
        num_warmup_steps=1000,
    )
    
    # Configure the optimizer
    initial_model.configure_optimizers()
    
    # Create a checkpoint dictionary with optimizer and scheduler states
    checkpoint = {
        "optimizer_states": [{"lr": initial_lr}],
        "lr_schedulers": [{"base_lrs": [initial_lr]}],
        "state_dict": initial_model.state_dict(),
    }
    
    # Create a new model with different parameters and override_optimizer_on_load=False
    new_lr = 1e-5
    model_without_override = LlamaLitModule(
        config=llama_config,
        tokenizer=mock_tokenizer,
        lr=new_lr,
        scheduler_name="linear",
        num_warmup_steps=500,
        override_optimizer_on_load=False,
    )
    
    # Load the checkpoint without override
    checkpoint_without_override = checkpoint.copy()
    model_without_override.on_load_checkpoint(checkpoint_without_override)
    
    # Check that optimizer and scheduler states are preserved
    assert "optimizer_states" in checkpoint_without_override
    assert "lr_schedulers" in checkpoint_without_override
    
    # Create a new model with different parameters and override_optimizer_on_load=True
    model_with_override = LlamaLitModule(
        config=llama_config,
        tokenizer=mock_tokenizer,
        lr=new_lr,
        scheduler_name="linear",
        num_warmup_steps=500,
        override_optimizer_on_load=True,
    )
    
    # Load the checkpoint with override
    checkpoint_with_override = checkpoint.copy()
    model_with_override.on_load_checkpoint(checkpoint_with_override)
    
    # Check that optimizer and scheduler states are removed
    assert "optimizer_states" not in checkpoint_with_override
    assert "lr_schedulers" not in checkpoint_with_override


def test_override_optimizer_integration(llama_config, mock_tokenizer):
    """Test the override_optimizer_on_load parameter is correctly passed through the class hierarchy."""
    # Test with override_optimizer_on_load=True
    lit_module = LlamaLitModule(
        config=llama_config,
        tokenizer=mock_tokenizer,
        override_optimizer_on_load=True,
    )
    
    # Check that the parameter was passed correctly
    assert lit_module.override_optimizer_on_load is True
    
    # Test with override_optimizer_on_load=False (default)
    lit_module = LlamaLitModule(
        config=llama_config,
        tokenizer=mock_tokenizer,
    )
    
    # Check that the parameter has the default value
    assert lit_module.override_optimizer_on_load is False