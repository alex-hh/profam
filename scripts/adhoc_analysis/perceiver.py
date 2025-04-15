from transformers import PerceiverModel, PerceiverConfig
from transformers.models.perceiver.modeling_perceiver import PerceiverClassificationDecoder, PerceiverAbstractDecoder
import torch
import torch.nn as nn


"""
Minimal working example of perceiver model,
As I understand it, we take a set of embeddings (eg. embedded MSA)
do cross-attention with a fixed number of learnable latent vectors.
Followed by multiple layers of self-attention on the latent vectors.
Due to fixed number of latents (unrelated to input size), memory and compute
does not scale quadratically with input size.
"""

# Create a custom class that combines PerceiverModel with a decoder for protein MSA processing
class ProteinMSAPerceiverModel(nn.Module):
    def __init__(self, vocab_size=21, embedding_dim=512, num_latents=128, latent_dim=512, output_dim=512):
        super().__init__()
        
        # Configure the Perceiver model
        self.config = PerceiverConfig(
            num_latents=num_latents,            # Number of latent vectors
            latent_dim=latent_dim,              # Size of latent vectors
            d_model=embedding_dim,              # Dimension of the model's hidden layers
            num_self_attends_per_block=6,       # Number of self-attention layers per block
            num_self_attention_heads=8,         # Number of attention heads for self-attention
            num_cross_attention_heads=8,        # Number of attention heads for cross-attention
            attention_probs_dropout_prob=0.1,   # Dropout probability for attention
            hidden_dropout_prob=0.1,            # Dropout probability for hidden layers
            vocab_size=vocab_size               # Size of vocabulary for MSA tokens
        )
        
        # Create the token embedding layer
        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Create the main Perceiver model
        self.perceiver = PerceiverModel(self.config)
        
        
    def forward(self, input_ids, attention_mask=None):
        # Convert token IDs to embeddings
        embeddings = self.token_embeddings(input_ids)
        
        # Process the input through the Perceiver
        perceiver_outputs = self.perceiver(
            inputs=embeddings,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # The latent vectors from Perceiver
        latent_vectors = perceiver_outputs.last_hidden_state

        return latent_vectors
# Create the model
model = ProteinMSAPerceiverModel(
    vocab_size=21,        # Assuming standard protein alphabet plus gap
    embedding_dim=512,    # Size of the token embeddings
    num_latents=128,      # Number of latent vectors
    latent_dim=512,       # Size of latent vectors
    output_dim=512        # Size of the final embedding
)

# Example: [batch_size, sequence_length]
input_ids = torch.randint(0, 21, (1, 333))  # Tokenized MSA (e.g., 0-20 for amino acids)
attention_mask = torch.ones_like(input_ids)  # All tokens are valid

# Forward pass
outputs = model(input_ids, attention_mask)

print(f"Input shape: {input_ids.shape}")
print(f"Output latent vectors shape: {outputs.shape}")
