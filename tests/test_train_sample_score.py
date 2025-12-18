import os

# Set environment variable for deterministic algorithms with CuBLAS
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
from hydra import compose, initialize_config_dir

from src.train import train

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def require_cuda():
    """Skip the entire module if a CUDA device is not available."""
    if not torch.cuda.is_available():
        pytest.skip("GPU is required for integration tests.", allow_module_level=True)


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


def test_training_on_example_data(tmp_path, project_root):
    run_dir = tmp_path / "train_run"
    with initialize_config_dir(
        config_dir=str(project_root / "configs"), version_base="1.3"
    ):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=[
                "experiment=train_profam_example",
                # lightweight model (previously `model=llama_nano`)
                "model.scheduler_name=inverse_sqrt",
                "model.lr=1e-3",
                "model.config.hidden_size=128",
                "model.config.intermediate_size=512",
                "model.config.num_attention_heads=2",
                "model.config.num_hidden_layers=5",
                "model.config.num_key_value_heads=2",
                "model.config.max_position_embeddings=8192",
                "model.config.scoring_max_tokens=10240",
                "model.config.attn_implementation=null",
                # keep the Trainer single-process for tests
                "trainer.timeout=null",
                "trainer.strategy=auto",
                "trainer.devices=1",
                # disable callbacks/loggers for a minimal integration run
                "callbacks=null",
                "logger=null",
                f"paths.root_dir={project_root}",
                f"paths.data_dir={project_root/'data'/'train_example'}",
                f"paths.output_dir={run_dir}",
                f"paths.log_dir={run_dir/'logs'}",
                f"hydra.run.dir={run_dir}",
                "hydra.output_subdir=null",
                "hydra/job_logging=disabled",
                "hydra/hydra_logging=disabled",
                "trainer.max_steps=1",
                "+trainer.limit_train_batches=1",
                "trainer.limit_val_batches=0",
                "trainer.val_check_interval=1",
                "trainer.log_every_n_steps=1",
                "+trainer.enable_checkpointing=False",
                "trainer.deterministic=True",
                "data.num_workers=4",
                "data.batch_size=2",
                "data.interleaved=False",
                "data.prefetch_factor=2",
                "train=True",
                "test=False",
            ],
        )
    metric_dict, obj_dict = train(cfg)
    trainer = obj_dict["trainer"]
    assert trainer.global_step >= 1, "Trainer did not run a training step on GPU"
    assert metric_dict is not None, "Training metrics were not produced"


def test_generate_sequences(tmp_path, project_root):
    ckpt_dir = project_root / "model_checkpoints" / "profam-1"
    fasta_path = (
        project_root
        / "data"
        / "generate_sequences_example"
        / "generate_sequences_test_case.fasta"
    )
    save_dir = tmp_path / "gen_outputs"
    save_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "scripts/generate_sequences.py",
        "--checkpoint_dir",
        str(ckpt_dir),
        "--file_path",
        str(fasta_path),
        "--save_dir",
        str(save_dir),
        "--sampler",
        "single",
        "--num_samples",
        "1",
        "--max_tokens",
        "2048",
        "--max_generated_length",
        "128",
        "--device",
        "cuda",
        "--dtype",
        "bfloat16",
        "--top_p",
        "0.9",
        "--attn_implementation",
        "sdpa",
    ]

    result = subprocess.run(
        cmd,
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Generation script failed: {result.stderr}"

    outputs = list(save_dir.glob("*generated_single.fasta"))
    assert outputs, "No generated FASTA file was produced"
    assert outputs[0].stat().st_size > 0, "Generated FASTA file is empty"


def test_score_sequences(tmp_path, project_root):
    ckpt_dir = project_root / "model_checkpoints" / "profam-1"
    conditioning = (
        project_root / "data" / "score_sequences_example" / "CCDB_ECOLI_Adkar_2012.a3m"
    )
    candidates = (
        project_root / "data" / "score_sequences_example" / "CCDB_ECOLI_Adkar_2012.csv"
    )
    save_dir = tmp_path / "score_outputs"
    save_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "scripts/score_sequences.py",
        "--checkpoint_dir",
        str(ckpt_dir),
        "--conditioning_fasta",
        str(conditioning),
        "--candidates_file",
        str(candidates),
        "--save_dir",
        str(save_dir),
        "--device",
        "cuda",
        "--dtype",
        "bfloat16",
        "--max_tokens",
        "2048",
        "--ensemble_number",
        "1",
        "--attn_implementation",
        "sdpa",
    ]

    result = subprocess.run(
        cmd,
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Scoring script failed: {result.stderr}"

    csv_path = save_dir / "CCDB_ECOLI_Adkar_2012_scores.csv"
    assert csv_path.exists(), "Score CSV was not created"
    df = pd.read_csv(csv_path)
    assert len(df) > 0, "Score CSV is empty"
    assert {"id", "mutated_sequence", "score"}.issubset(
        df.columns
    ), "Score CSV missing expected columns"
