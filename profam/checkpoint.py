"""Shared model loading logic for ProFam checkpoints."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from profam.constants import resolve_runtime_path
from profam.models.llama import LlamaLitModule

_DEFAULT_CHECKPOINT = "model_checkpoints/profam-1/checkpoints/last.ckpt"


def load_model(
    checkpoint: str | os.PathLike = _DEFAULT_CHECKPOINT,
    device: str | None = None,
    dtype: str = "bfloat16",
    attn_implementation: str = "sdpa",
    auto_download: bool = True,
) -> LlamaLitModule:
    """Load a ProFam model from a Lightning checkpoint.

    Parameters
    ----------
    checkpoint:
        Path to a ``.ckpt`` file.  Relative paths are resolved against the
        current working directory and the repository root.
    device:
        Target device (e.g. ``"cuda"`` or ``"cpu"``).  *None* auto-detects.
    dtype:
        One of ``"float32"``, ``"float16"``, ``"bfloat16"``.
    attn_implementation:
        Attention backend — ``"sdpa"``, ``"flash_attention_2"``, or ``"eager"``.
    auto_download:
        When *True* and the resolved checkpoint path does not exist, attempt
        to download the default ProFam-1 checkpoint before loading.

    Returns
    -------
    LlamaLitModule
        The model in eval mode on the requested device/dtype.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = resolve_runtime_path(checkpoint)

    if not ckpt_path.exists() and auto_download:
        # Only auto-download when using the default checkpoint pattern
        if str(checkpoint) == _DEFAULT_CHECKPOINT or str(ckpt_path).endswith(
            "model_checkpoints/profam-1/checkpoints/last.ckpt"
        ):
            from profam.download_checkpoint import download_checkpoint

            print("Checkpoint not found — downloading ProFam-1 …")
            download_checkpoint()
            ckpt_path = resolve_runtime_path(checkpoint)

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. "
            "Run `profam download` or `profam-download-checkpoint` to download it."
        )

    # Validate flash-attention availability
    if attn_implementation == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            raise ImportError(
                "Flash attention is not installed. Select an alternative attention "
                "implementation such as `--attn_implementation sdpa`, or install it "
                "with `pip install flash-attn --no-build-isolation`."
            )

    try:
        ckpt_blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hyper_params = ckpt_blob.get("hyper_parameters", {})
        cfg_obj = hyper_params.get("config", None)
        if cfg_obj is None:
            raise RuntimeError(
                "Could not find 'config' in checkpoint hyper_parameters "
                "to override attention implementation"
            )
        setattr(cfg_obj, "attn_implementation", attn_implementation)
        setattr(cfg_obj, "_attn_implementation", attn_implementation)
        # ProFam-1 was trained with llama3 RoPE scaling.  The checkpoint
        # stores rope_scaling as an OmegaConf DictConfig (from Hydra).
        # Transformers >=4.49 changed ``LlamaRotaryEmbedding.__init__`` to
        # use ``isinstance(config.rope_scaling, dict)`` instead of
        # ``config.rope_scaling is not None``, so a DictConfig silently
        # falls back to the wrong "default" RoPE — producing corrupted
        # attention for long sequences.  We convert to a plain dict and
        # explicitly ensure the rope_type key is present.
        rs = getattr(cfg_obj, "rope_scaling", None)
        if rs is not None:
            try:
                from omegaconf import OmegaConf

                rs = OmegaConf.to_container(rs, resolve=True)
            except Exception:
                rs = dict(rs)
            rs.setdefault("rope_type", "llama3")
            cfg_obj.rope_scaling = rs
        model: LlamaLitModule = LlamaLitModule.load_from_checkpoint(
            str(ckpt_path), config=cfg_obj, strict=False, weights_only=False
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")

    model.eval()
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    model.to(device, dtype=dtype_map[dtype])
    return model
