"""End-to-end integration test for the ``profam`` package.

Installs the local source tree into a fresh virtual environment (the
same code path ``pip install profam`` would exercise) and drives each of
the three public entry points against the bundled CCDB_ECOLI score
example:

  * Python API       (``from profam import ProFam``)
  * ``profam`` CLI   (``profam score`` / ``profam generate``)
  * Legacy scripts   (``scripts/score_sequences.py`` /
                      ``scripts/generate_sequences.py``)

The three entry points must agree on the scores they assign and on the
sequences they generate. Scoring is also exercised under the three
regimes requested by the spec (aligned + ensemble=2 with diversity
weighting, unaligned + ensemble=2 without diversity weighting, and
no-context) and asserted to fall within pre-measured Spearman ranges.
Finally, a training smoke test resumes from the ProFam-1 checkpoint and
takes a small number of optimizer steps on the example training data.

Requires a CUDA device and the ProFam-1 checkpoint at
``model_checkpoints/profam-1/checkpoints/last.ckpt``. The venv install
takes roughly 30 s on a warm ``uv`` cache, so the test is marked slow.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import spearmanr

pytestmark = pytest.mark.slow

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CKPT_REL = "model_checkpoints/profam-1/checkpoints/last.ckpt"
CKPT_DIR_REL = "model_checkpoints/profam-1"
A3M_REL = "data/score_sequences_example/CCDB_ECOLI_Adkar_2012.a3m"
CSV_REL = "data/score_sequences_example/CCDB_ECOLI_Adkar_2012_subsample_250.csv"
GEN_FASTA_REL = "data/generate_sequences_example/generate_sequences_test_case.fasta"

# Scoring is deterministic on this fixture (the ensemble sampler pins its
# own RNG seed internally) so the observed Spearman values were identical
# across three seeds during measurement. We widen each interval by ±0.05
# to tolerate minor cuBLAS / dtype drift on different GPUs.
SAFETY_MARGIN = 0.05
EXPECTED_SPEARMAN = {
    # regime_name: (measured_value, (lo, hi))
    "aligned_ensemble2": (0.4562, (0.4562 - SAFETY_MARGIN, 0.4562 + SAFETY_MARGIN)),
    "unaligned_ensemble2": (0.3073, (0.3073 - SAFETY_MARGIN, 0.3073 + SAFETY_MARGIN)),
    "no_context": (0.1080, (0.1080 - SAFETY_MARGIN, 0.1080 + SAFETY_MARGIN)),
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for integration tests", allow_module_level=True)


@pytest.fixture(scope="module", autouse=True)
def _require_checkpoint() -> None:
    ckpt = PROJECT_ROOT / CKPT_REL
    if not ckpt.exists():
        pytest.skip(
            f"ProFam-1 checkpoint missing at {ckpt}; run `profam download` first.",
            allow_module_level=True,
        )


@pytest.fixture(scope="module")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="module")
def installed_venv(tmp_path_factory, project_root) -> Path:
    """Build a fresh venv and install the local package into it.

    Uses ``uv sync --frozen`` against the repo's lockfile so the test
    runs against a reproducible dependency set (unlocked resolution can
    pick a PyTorch build that demands a newer NVIDIA driver than the CI
    host has). The venv lives in a temp dir distinct from the dev
    ``./.venv``, and the project is installed non-editable so every
    entry point is resolved through the site-packages copy rather than
    the source tree.
    """
    if shutil.which("uv") is None:
        pytest.skip("`uv` is required to build the integration venv")

    venv_dir = tmp_path_factory.mktemp("profam_integration") / "venv"
    env = os.environ.copy()
    env["UV_PROJECT_ENVIRONMENT"] = str(venv_dir)
    subprocess.run(
        [
            "uv",
            "sync",
            "--frozen",
            "--no-dev",
            "--no-editable",
            # Force a fresh wheel build from the current source tree;
            # without this, uv's archive cache happily reuses an older
            # wheel built under the same version number and test runs
            # won't reflect uncommitted edits to the local package.
            "--reinstall-package",
            "profam",
            "--project",
            str(project_root),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    venv_python = venv_dir / "bin" / "python"
    # sanity-check the public API import survived the install
    subprocess.run(
        [str(venv_python), "-c", "from profam import ProFam"],
        check=True,
        capture_output=True,
        text=True,
    )
    return venv_python


@pytest.fixture(scope="module")
def candidates_df(project_root) -> pd.DataFrame:
    return pd.read_csv(project_root / CSV_REL)


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


def _run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    """Run a command and surface stdout/stderr on failure."""
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Command failed (rc={result.returncode}):\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stdout:\n{result.stdout}\n"
            f"  stderr:\n{result.stderr}"
        )


def _api_score_script(
    project_root: Path,
    out_npy: Path,
    prompt_kind: str,  # "aligned" | "unaligned" | "none"
    ensemble_size: int,
    use_diversity_weights: bool,
) -> str:
    """Build a self-contained Python snippet that runs ProFam.score."""
    ckpt = str(project_root / CKPT_REL)
    csv = str(project_root / CSV_REL)
    a3m = str(project_root / A3M_REL)
    return textwrap.dedent(
        f"""
        import numpy as np
        import pandas as pd
        from profam import ProFam

        model = ProFam(checkpoint={ckpt!r}, device='cuda', dtype='bfloat16',
                       attn_implementation='sdpa', auto_download=False)
        df = pd.read_csv({csv!r})
        cand = df['mutated_sequence'].astype(str).str.upper().tolist()

        kind = {prompt_kind!r}
        if kind in ('aligned', 'unaligned'):
            prompt = {a3m!r}
        else:
            prompt = None

        res = model.score(
            sequences=cand,
            prompt=prompt,
            ensemble_size={ensemble_size},
            use_diversity_weights={use_diversity_weights},
        )
        np.save({str(out_npy)!r}, np.asarray(res.scores))
        """
    )


def _api_generate_script(
    project_root: Path,
    out_json: Path,
    num_samples: int,
    seed: int,
) -> str:
    ckpt = str(project_root / CKPT_REL)
    fasta = str(project_root / GEN_FASTA_REL)
    return textwrap.dedent(
        f"""
        import json
        from profam import ProFam
        from profam.sequence.fasta import read_fasta

        model = ProFam(checkpoint={ckpt!r}, device='cuda', dtype='bfloat16',
                       attn_implementation='sdpa', auto_download=False)
        accessions, sequences = read_fasta({fasta!r}, keep_insertions=True,
                                           to_upper=True, keep_gaps=False)
        res = model.generate(
            prompt=sequences,
            prompt_accessions=accessions,
            num_samples={num_samples},
            max_tokens=2048,
            max_generated_length=128,
            top_p=0.9,
            sampler='single',
            seed={seed},
        )
        with open({str(out_json)!r}, 'w') as f:
            json.dump({{'sequences': res.sequences,
                        'scores':    [float(s) for s in res.scores]}}, f)
        """
    )


def _read_scores_csv(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    assert {"id", "mutated_sequence", "score"}.issubset(df.columns)
    return df["score"].to_numpy()


def _read_generated_fasta(path: Path) -> list[str]:
    sequences: list[str] = []
    current: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if current:
                sequences.append("".join(current))
                current = []
        else:
            current.append(line.strip())
    if current:
        sequences.append("".join(current))
    return sequences


# ---------------------------------------------------------------------------
# Scoring parity: API == CLI == legacy
# ---------------------------------------------------------------------------


def test_score_api_cli_legacy_parity(
    tmp_path, project_root, installed_venv, candidates_df
):
    """The three entry points must produce the same ensemble-3 scores."""
    api_dir = tmp_path / "api"
    cli_dir = tmp_path / "cli"
    legacy_dir = tmp_path / "legacy"
    for d in (api_dir, cli_dir, legacy_dir):
        d.mkdir()

    # --- Python API -------------------------------------------------------
    api_npy = api_dir / "scores.npy"
    _run(
        [
            str(installed_venv),
            "-c",
            _api_score_script(
                project_root=project_root,
                out_npy=api_npy,
                prompt_kind="aligned",
                ensemble_size=3,
                use_diversity_weights=True,
            ),
        ]
    )
    api_scores = np.load(api_npy)

    # --- CLI ``profam score`` ---------------------------------------------
    _run(
        [
            str(installed_venv.parent / "profam"),
            "score",
            "--checkpoint",
            str(project_root / CKPT_REL),
            "--conditioning_fasta",
            str(project_root / A3M_REL),
            "--candidates_file",
            str(project_root / CSV_REL),
            "--save_dir",
            str(cli_dir),
            "--ensemble_number",
            "3",
            "--use_diversity_weights",
            "--device",
            "cuda",
            "--dtype",
            "bfloat16",
            "--attn_implementation",
            "sdpa",
            # avoid polluting the bundled weights cache next to the MSA
            "--recompute_diversity_weights",
        ]
    )
    cli_scores = _read_scores_csv(
        cli_dir / "CCDB_ECOLI_Adkar_2012_subsample_250_scores.csv"
    )

    # --- Legacy script ----------------------------------------------------
    _run(
        [
            str(installed_venv),
            str(project_root / "scripts" / "score_sequences.py"),
            "--checkpoint_dir",
            str(project_root / CKPT_DIR_REL),
            "--conditioning_fasta",
            str(project_root / A3M_REL),
            "--candidates_file",
            str(project_root / CSV_REL),
            "--save_dir",
            str(legacy_dir),
            "--ensemble_number",
            "3",
            "--use_diversity_weights",
            "--device",
            "cuda",
            "--dtype",
            "bfloat16",
            "--attn_implementation",
            "sdpa",
            "--recompute_diversity_weights",
        ],
        cwd=project_root,
    )
    legacy_scores = _read_scores_csv(
        legacy_dir / "CCDB_ECOLI_Adkar_2012_subsample_250_scores.csv"
    )

    assert (
        api_scores.shape
        == cli_scores.shape
        == legacy_scores.shape
        == (len(candidates_df),)
    ), (
        f"shape mismatch: api={api_scores.shape}, cli={cli_scores.shape}, "
        f"legacy={legacy_scores.shape}"
    )

    # Scoring pins its own RNG; expect bfloat16-level agreement across entry points.
    np.testing.assert_allclose(cli_scores, api_scores, atol=5e-3, rtol=0)
    np.testing.assert_allclose(legacy_scores, api_scores, atol=5e-3, rtol=0)


# ---------------------------------------------------------------------------
# Generation parity: API == CLI == legacy
# ---------------------------------------------------------------------------


def test_generate_api_cli_legacy_parity(tmp_path, project_root, installed_venv):
    """With a pinned seed, the three entry points must produce the same sequences."""
    api_dir = tmp_path / "api"
    cli_dir = tmp_path / "cli"
    legacy_dir = tmp_path / "legacy"
    for d in (api_dir, cli_dir, legacy_dir):
        d.mkdir()

    seed = 42
    num_samples = 2

    # --- Python API -------------------------------------------------------
    api_json = api_dir / "gen.json"
    _run(
        [
            str(installed_venv),
            "-c",
            _api_generate_script(
                project_root=project_root,
                out_json=api_json,
                num_samples=num_samples,
                seed=seed,
            ),
        ]
    )
    api_payload = json.loads(api_json.read_text())
    api_sequences = api_payload["sequences"]

    common_flags = [
        "--file_path",
        str(project_root / GEN_FASTA_REL),
        "--num_samples",
        str(num_samples),
        "--max_tokens",
        "2048",
        "--max_generated_length",
        "128",
        "--sampler",
        "single",
        "--top_p",
        "0.9",
        "--seed",
        str(seed),
        "--device",
        "cuda",
        "--dtype",
        "bfloat16",
        "--attn_implementation",
        "sdpa",
    ]

    # --- CLI ``profam generate`` ------------------------------------------
    _run(
        [
            str(installed_venv.parent / "profam"),
            "generate",
            "--checkpoint",
            str(project_root / CKPT_REL),
            "--save_dir",
            str(cli_dir),
            "--no-auto_download",
            *common_flags,
        ]
    )
    cli_sequences = _read_generated_fasta(
        cli_dir / "generate_sequences_test_case_generated_single.fasta"
    )

    # --- Legacy script ----------------------------------------------------
    _run(
        [
            str(installed_venv),
            str(project_root / "scripts" / "generate_sequences.py"),
            "--checkpoint_dir",
            str(project_root / CKPT_DIR_REL),
            "--save_dir",
            str(legacy_dir),
            *common_flags,
        ],
        cwd=project_root,
    )
    legacy_sequences = _read_generated_fasta(
        legacy_dir / "generate_sequences_test_case_generated_single.fasta"
    )

    assert (
        len(api_sequences) == len(cli_sequences) == len(legacy_sequences) == num_samples
    )
    # Same seed + same model + same prompt must produce bit-identical strings.
    assert api_sequences == cli_sequences
    assert api_sequences == legacy_sequences


# ---------------------------------------------------------------------------
# Spearman bounds across three scoring regimes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "regime,prompt_kind,ensemble_size,use_diversity_weights",
    [
        ("aligned_ensemble2", "aligned", 2, True),
        ("unaligned_ensemble2", "unaligned", 2, False),
        ("no_context", "none", 1, False),
    ],
)
def test_scoring_spearman_in_expected_range(
    tmp_path,
    project_root,
    installed_venv,
    candidates_df,
    regime,
    prompt_kind,
    ensemble_size,
    use_diversity_weights,
):
    out_npy = tmp_path / f"scores_{regime}.npy"
    _run(
        [
            str(installed_venv),
            "-c",
            _api_score_script(
                project_root=project_root,
                out_npy=out_npy,
                prompt_kind=prompt_kind,
                ensemble_size=ensemble_size,
                use_diversity_weights=use_diversity_weights,
            ),
        ]
    )
    scores = np.load(out_npy)
    dms = candidates_df["DMS_score"].to_numpy()
    rho, _ = spearmanr(scores, dms)

    _, (lo, hi) = EXPECTED_SPEARMAN[regime]
    assert lo <= rho <= hi, (
        f"Spearman for regime {regime!r} outside expected range: "
        f"got {rho:.4f}, expected in [{lo:.4f}, {hi:.4f}] "
        f"(measured baseline {EXPECTED_SPEARMAN[regime][0]:.4f} ± {SAFETY_MARGIN})"
    )


# ---------------------------------------------------------------------------
# Training smoke test: resume from checkpoint, take a few steps
# ---------------------------------------------------------------------------


def test_training_resumes_from_checkpoint_and_takes_steps(
    tmp_path, project_root, installed_venv
):
    """Resume training from the ProFam-1 checkpoint and take a few steps.

    Exercises the full Hydra + Lightning training path under the
    installed package, confirming that the shipped checkpoint is
    loadable for continued training and that the optimizer actually
    advances. The checkpoint is loaded via ``profam.checkpoint.load_model``
    (the public path) rather than Lightning's built-in ``ckpt_path`` —
    Lightning's path tries ``weights_only=True`` + strict state-dict
    loading, neither of which the shipped checkpoint satisfies.
    """
    run_dir = tmp_path / "train_run"
    run_dir.mkdir()

    ckpt = project_root / CKPT_REL

    overrides = [
        "experiment=train_profam_example",
        "trainer.timeout=null",
        "trainer.strategy=auto",
        "trainer.devices=1",
        "trainer.precision=bf16-true",
        "callbacks=null",
        "logger=null",
        f"paths.root_dir={project_root}",
        f"paths.data_dir={project_root / 'data' / 'train_example'}",
        f"paths.output_dir={run_dir}",
        f"paths.log_dir={run_dir / 'logs'}",
        f"hydra.run.dir={run_dir}",
        "hydra.output_subdir=null",
        "hydra/job_logging=disabled",
        "hydra/hydra_logging=disabled",
        "trainer.max_steps=3",
        "+trainer.limit_train_batches=3",
        "trainer.limit_val_batches=0",
        "trainer.val_check_interval=1",
        "trainer.log_every_n_steps=1",
        "+trainer.enable_checkpointing=False",
        "trainer.deterministic=False",
        "data.num_workers=2",
        "data.batch_size=1",
        "data.interleaved=False",
        "data.prefetch_factor=2",
    ]

    driver = textwrap.dedent(
        f"""
        import json
        import os

        import hydra
        import lightning as L
        from hydra import compose, initialize_config_dir

        from profam.checkpoint import load_model
        from profam.constants import CONFIGS_DIR
        from profam.utils import RankedLogger, setup_profiler

        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        # Load the real checkpoint via the supported path (handles
        # weights_only=False + strict=False), then run a short training
        # loop on top of it with a Hydra-composed datamodule + trainer.
        model = load_model(
            checkpoint={str(ckpt)!r},
            device='cuda',
            dtype='bfloat16',
            attn_implementation='sdpa',
            auto_download=False,
        )
        start_step = int(model.global_step) if hasattr(model, 'global_step') else 0

        with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base='1.3'):
            cfg = compose(
                config_name='train.yaml',
                return_hydra_config=True,
                overrides={overrides!r},
            )

        L.seed_everything(cfg.get('seed', 12345), workers=True)

        datamodule = hydra.utils.instantiate(
            cfg.data, tokenizer=model.tokenizer, _convert_='partial'
        )
        profiler = setup_profiler(
            cfg=cfg.trainer.profiler,
            log=RankedLogger(__name__, rank_zero_only=True),
        )
        trainer = hydra.utils.instantiate(
            cfg.trainer, callbacks=None, logger=None, profiler=profiler
        )

        trainer.fit(model=model, datamodule=datamodule)

        with open({str(run_dir / 'result.json')!r}, 'w') as f:
            json.dump({{
                'global_step_before': start_step,
                'global_step_after': int(trainer.global_step),
                'max_steps': int(cfg.trainer.max_steps),
            }}, f)
        """
    )

    _run([str(installed_venv), "-c", driver], cwd=project_root)

    result = json.loads((run_dir / "result.json").read_text())
    assert result["global_step_after"] >= result["max_steps"], (
        f"trainer did not complete {result['max_steps']} steps from the "
        f"loaded checkpoint: global_step_after={result['global_step_after']}"
    )
