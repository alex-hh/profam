from typing import Optional

from transformers import LlamaConfig, PreTrainedTokenizerFast

from src.models.diffusion.gaussian_diffusion import GaussianDiffusion
from src.models.diffusion.profusion import ProFusionLitModule
from src.models.diffusion.wrapper import WrappedHFProFusionModel


class LlamaProfusion(ProFusionLitModule):
    def __init__(
        self,
        config: LlamaConfig,
        diffusion: GaussianDiffusion,
        tokenizer: PreTrainedTokenizerFast,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        scheduler_name: Optional[str] = None,
        num_warmup_steps: int = 1000,
        num_training_steps: Optional[int] = None,
        scoring_max_tokens: int = 10240,
        use_kv_cache_for_scoring: bool = True,
        num_atoms: int = 1,
        diffusion_loss_weight: float = 1.0,
        diffusion_loss_prob: float = 1.0,
    ) -> None:
        model = WrappedHFProFusionModel(
            config,
            token_embedder="model.embed_tokens",
            embedding_dim=config.hidden_size,
            use_seq_pos=tokenizer.use_seq_pos,
            max_seq_pos=tokenizer.max_seq_pos,
            num_atoms=num_atoms,
            num_timesteps=diffusion.original_num_timesteps,  # assumes a spacediffusion instance actually
        )
        super().__init__(
            model,
            diffusion,
            tokenizer,
            lr=lr,
            weight_decay=weight_decay,
            scheduler_name=scheduler_name,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            scoring_max_tokens=scoring_max_tokens,
            use_kv_cache_for_scoring=use_kv_cache_for_scoring,
            diffusion_loss_weight=diffusion_loss_weight,
            diffusion_loss_prob=diffusion_loss_prob,
        )
