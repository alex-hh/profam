"""Tests for the public ProFam Python API."""

import numpy as np
import pytest
import torch

from profam.api import (
    ConditioningPrompt,
    GenerationResult,
    ProFam,
    ScoringResult,
    _build_protein_document,
    _resolve_prompt,
)
from profam.cli.score_sequences import _load_conditioning_views
from profam.data.objects import ProteinDocument


class TestResolvePrompt:
    def test_none_passes_through(self):
        conditioning, aligned, source_path = _resolve_prompt(
            None, use_diversity_weights=False
        )
        assert conditioning == []
        assert aligned is None
        assert source_path is None

    def test_aligned_list_derives_both_views(self):
        conditioning, aligned, source_path = _resolve_prompt(
            ["ACD-E", "ACD-E", "AC--E"], use_diversity_weights=True
        )
        assert aligned == ["ACD-E", "ACD-E", "AC--E"]
        assert conditioning == ["ACDE", "ACDE", "ACE"]
        assert source_path is None

    def test_aligned_list_with_a3m_insertions(self):
        # a2m/a3m-style: lowercase letters and '.' are insertions; after
        # stripping them every row must have the same length.
        conditioning, aligned, _ = _resolve_prompt(
            ["ACD-E", "ACDaa-E", "ACD..-E"], use_diversity_weights=True
        )
        assert aligned == ["ACD-E", "ACD-E", "ACD-E"]
        assert conditioning == ["ACDE", "ACDAAE", "ACDE"]

    def test_unaligned_list_with_weights_raises(self):
        with pytest.raises(ValueError, match="aligned"):
            _resolve_prompt(["ACDE", "ACD"], use_diversity_weights=True)

    def test_unaligned_list_without_weights_returns_none_aligned(self):
        conditioning, aligned, _ = _resolve_prompt(
            ["ACDE", "ACDEF", "AC"], use_diversity_weights=False
        )
        assert aligned is None
        assert conditioning == ["ACDE", "ACDEF", "AC"]

    def test_aligned_file(self, tmp_path):
        path = tmp_path / "tiny.a3m"
        path.write_text(">s1\nACDE\n>s2\nACDaE\n>s3\nAC-E\n")
        conditioning, aligned, source_path = _resolve_prompt(
            str(path), use_diversity_weights=True
        )
        assert aligned is not None
        assert {len(s) for s in aligned} == {4}
        assert conditioning == ["ACDE", "ACDAE", "ACE"]
        assert source_path == str(path)

    def test_ragged_file_with_weights_raises(self, tmp_path):
        path = tmp_path / "ragged.fasta"
        path.write_text(">s1\nACDE\n>s2\nACDEF\n")
        with pytest.raises(ValueError, match="aligned"):
            _resolve_prompt(str(path), use_diversity_weights=True)

    def test_ragged_file_without_weights_falls_back(self, tmp_path):
        path = tmp_path / "ragged.fasta"
        path.write_text(">s1\nACDE\n>s2\nACDEF\n")
        conditioning, aligned, source_path = _resolve_prompt(
            str(path), use_diversity_weights=False
        )
        assert aligned is None
        assert conditioning == ["ACDE", "ACDEF"]
        assert source_path == str(path)

    def test_bad_type_raises(self):
        with pytest.raises(TypeError):
            _resolve_prompt(42, use_diversity_weights=False)


class TestCliLoadConditioningViews:
    def _write_aligned(self, tmp_path):
        path = tmp_path / "msa.a3m"
        path.write_text(">s1\nACDE\n>s2\nACDaE\n>s3\nAC-E\n")
        return path

    def _write_ragged(self, tmp_path):
        path = tmp_path / "ragged.fasta"
        path.write_text(">s1\nACDE\n>s2\nACDEFGHI\n>s3\nACD\n")
        return path

    def test_returns_aligned_views_for_msa(self, tmp_path):
        path = self._write_aligned(tmp_path)
        conditioning, aligned = _load_conditioning_views(
            str(path), use_diversity_weights=True
        )
        assert aligned is not None
        assert len(conditioning) == 3

    def test_falls_back_to_unaligned_when_weights_disabled(self, tmp_path, capsys):
        path = self._write_ragged(tmp_path)
        conditioning, aligned = _load_conditioning_views(
            str(path), use_diversity_weights=False
        )
        assert aligned is None
        assert len(conditioning) == 3
        err = capsys.readouterr().err
        assert "not an aligned MSA" in err
        assert "diversity weights disabled" in err

    def test_raises_when_ragged_and_weights_requested(self, tmp_path):
        path = self._write_ragged(tmp_path)
        with pytest.raises(SystemExit) as excinfo:
            _load_conditioning_views(str(path), use_diversity_weights=True)
        # SystemExit carries the error string as its code when non-int.
        msg = str(excinfo.value)
        assert "--no-use_diversity_weights" in msg
        assert "aligned MSA" in msg


def test_build_protein_document_from_strings():
    seqs = ["ACDEFGHIK", "LMNPQRST"]
    doc = _build_protein_document(seqs)
    assert isinstance(doc, ProteinDocument)
    assert list(doc.sequences) == seqs
    assert len(doc.accessions) == 2
    assert doc.accessions == ["seq_0", "seq_1"]


def test_build_protein_document_preserves_accessions():
    seqs = ["ACDEFGHIK", "LMNPQRST"]
    accs = ["sp|P1|FOO", "sp|Q2|BAR"]
    doc = _build_protein_document(seqs, accessions=accs)
    assert list(doc.accessions) == accs
    assert doc.representative_accession == accs[0]


def test_build_protein_document_rejects_mismatched_accessions():
    with pytest.raises(ValueError, match="does not match"):
        _build_protein_document(["ACDE", "FGHI"], accessions=["only_one"])


def test_generation_result_structure():
    result = GenerationResult(sequences=["ACDE", "FGHI"], scores=[-1.0, -2.0])
    assert len(result.sequences) == 2
    assert len(result.scores) == 2
    assert result.conditioning_prompts is None


def test_generation_result_with_conditioning_prompts():
    result = GenerationResult(
        sequences=["ACDE"],
        scores=[-1.0],
        conditioning_prompts=[
            ConditioningPrompt(sequences=["ACDEFGHIK"], accessions=["sp|P1|FOO"]),
        ],
    )
    assert result.conditioning_prompts is not None
    assert len(result.conditioning_prompts) == 1
    assert result.conditioning_prompts[0].sequences == ["ACDEFGHIK"]
    assert result.conditioning_prompts[0].accessions == ["sp|P1|FOO"]


def test_scoring_result_structure():
    scores = np.array([-1.0, -2.0])
    result = ScoringResult(sequences=["ACDE", "FGHI"], scores=scores)
    assert result.residue_scores is None
    np.testing.assert_array_equal(result.scores, scores)


def test_scoring_result_with_residue_scores():
    residue = [np.array([-0.1, -0.2, -0.3]), np.array([-0.4, -0.5])]
    result = ScoringResult(
        sequences=["ACD", "FG"],
        scores=np.array([-0.2, -0.45]),
        residue_scores=residue,
    )
    assert len(result.residue_scores) == 2
    assert result.residue_scores[0].shape == (3,)


class TestProFamWithTestModel:
    """Integration tests using the lightweight test model from conftest."""

    @pytest.fixture(autouse=True)
    def _setup(self, test_model):
        self.model = test_model
        self.model.eval()

    def _make_profam(self) -> ProFam:
        """Create a ProFam instance wrapping the test model (skip download)."""
        pf = object.__new__(ProFam)
        pf._model = self.model
        return pf

    def test_generate_returns_generation_result(self):
        pf = self._make_profam()
        result = pf.generate(
            prompt=["ACDEFGHIKLMNPQRSTVWY"],
            num_samples=1,
            max_tokens=2048,
            max_generated_length=10,
            seed=42,
        )
        assert isinstance(result, GenerationResult)
        assert len(result.sequences) == 1
        assert len(result.scores) == 1
        assert isinstance(result.sequences[0], str)
        assert result.conditioning_prompts is not None
        assert len(result.conditioning_prompts) == 1
        assert len(result.conditioning_prompts[0].sequences) >= 1
        assert len(result.conditioning_prompts[0].accessions) == len(
            result.conditioning_prompts[0].sequences
        )

    def test_generate_preserves_prompt_accessions_single(self):
        pf = self._make_profam()
        result = pf.generate(
            prompt=["ACDEFGHIKLMNPQRSTVWY", "WYVTSRQPNMLKIHGFEDCA"],
            prompt_accessions=["seqA", "seqB"],
            num_samples=1,
            max_tokens=2048,
            max_generated_length=5,
            seed=7,
            sampler="single",
        )
        assert result.conditioning_prompts is not None
        cond = result.conditioning_prompts[0]
        assert set(cond.accessions) <= {"seqA", "seqB"}
        for acc, seq in zip(cond.accessions, cond.sequences):
            if acc == "seqA":
                assert seq.upper() == "ACDEFGHIKLMNPQRSTVWY"
            elif acc == "seqB":
                assert seq.upper() == "WYVTSRQPNMLKIHGFEDCA"

    def test_generate_ensemble_returns_multiple_conditioning_prompts(self):
        pf = self._make_profam()
        result = pf.generate(
            prompt=[
                "ACDEFGHIKLMNPQRSTVWY",
                "WYVTSRQPNMLKIHGFEDCA",
                "ACDEFGHIKLMNPQRSTVWA",
                "MCDEFGHIKLMNPQRSTVWY",
            ],
            prompt_accessions=["a", "b", "c", "d"],
            num_samples=1,
            max_tokens=2048,
            max_generated_length=5,
            sampler="ensemble",
            num_prompts_in_ensemble=3,
            seed=11,
        )
        assert result.conditioning_prompts is not None
        assert len(result.conditioning_prompts) == 3
        for cond in result.conditioning_prompts:
            assert len(cond.accessions) == len(cond.sequences)
            assert all(a in {"a", "b", "c", "d"} for a in cond.accessions)

    def test_score_returns_scoring_result(self):
        pf = self._make_profam()
        result = pf.score(
            sequences=["ACDEFGHIKLMNPQRSTVWY"],
            prompt=["ACDEFGHIKLMNPQRSTVWY"],
            ensemble_size=1,
            max_tokens=2048,
            use_diversity_weights=False,
            seed=42,
        )
        assert isinstance(result, ScoringResult)
        assert len(result.sequences) == 1
        assert result.scores.shape == (1,)

    def test_score_no_context(self):
        pf = self._make_profam()
        result = pf.score(
            sequences=["ACDEFGHIKLMNPQRSTVWY"],
            prompt=None,
            seed=42,
        )
        assert isinstance(result, ScoringResult)
        assert result.scores.shape == (1,)

    def test_score_per_residue(self):
        pf = self._make_profam()
        result = pf.score(
            sequences=["ACDEFGHIKLMNPQRSTVWY"],
            prompt=["ACDEFGHIKLMNPQRSTVWY"],
            ensemble_size=1,
            max_tokens=2048,
            per_residue=True,
            use_diversity_weights=False,
            seed=42,
        )
        assert result.residue_scores is not None
        assert len(result.residue_scores) == 1
        assert isinstance(result.residue_scores[0], np.ndarray)
        assert len(result.residue_scores[0]) > 0

    def test_score_accepts_aligned_list(self):
        pf = self._make_profam()
        result = pf.score(
            sequences=["ACDEFGHIKLMNPQRSTVWY"],
            prompt=["ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIKLMNPQRSTVWY"],
            ensemble_size=1,
            max_tokens=2048,
            use_diversity_weights=True,
            seed=42,
        )
        assert isinstance(result, ScoringResult)
        assert result.scores.shape == (1,)

    def test_score_cache_weights_requires_path(self):
        pf = self._make_profam()
        with pytest.raises(ValueError, match="cache_weights=True requires"):
            pf.score(
                sequences=["ACDEFGHIKLMNPQRSTVWY"],
                prompt=["ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIKLMNPQRSTVWY"],
                ensemble_size=1,
                max_tokens=2048,
                use_diversity_weights=True,
                cache_weights=True,
                seed=42,
            )

    def test_score_cache_weights_uses_cache_for_file_prompt(self, tmp_path):
        pf = self._make_profam()
        a3m = tmp_path / "tiny.a3m"
        # Two aligned sequences so homology weights are well-defined.
        a3m.write_text(">s1\nACDEFGHIKLMNPQRSTVWY\n>s2\nACDEFGHIKLMNPQRSTVWY\n")

        result = pf.score(
            sequences=["ACDEFGHIKLMNPQRSTVWY"],
            prompt=str(a3m),
            ensemble_size=1,
            max_tokens=2048,
            use_diversity_weights=True,
            cache_weights=True,
            seed=42,
        )
        assert result.scores.shape == (1,)
        # Cache file should be written next to the MSA file.
        cache_path = a3m.with_name(a3m.stem + "_weights.npz")
        assert cache_path.exists()

    def test_score_unaligned_list_with_weights_raises(self):
        pf = self._make_profam()
        with pytest.raises(ValueError, match="aligned"):
            pf.score(
                sequences=["ACDEFGHIKLMNPQRSTVWY"],
                prompt=["ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIKLMNPQR"],
                ensemble_size=1,
                max_tokens=2048,
                use_diversity_weights=True,
                seed=42,
            )

    def test_score_seqs_per_residue_matches_legacy_compute_residue_scores(self):
        """score_seqs(..., return_per_residue=True) must match the legacy
        _compute_residue_scores logic (single-prompt, no-cache forward pass)
        position-by-position for the conditioned case.
        """
        from profam.models.utils import log_likelihood_from_outputs

        pf = self._make_profam()
        model = pf._model

        sequences = [
            "ACDEFGHIKLMNPQRSTVWY",
            "ACDEFGHIKLMNPQRSTVWYAC",
        ]
        prompt_seq = "ACDEFGHIKLMNPQRSTVWY"

        comp_tok = model.tokenizer.encode_completions(
            sequences,
            bos_token=model.tokenizer.sep_token,
            eos_token=model.tokenizer.sep_token,
        )
        completion_ids = (
            torch.as_tensor(comp_tok["input_ids"], dtype=torch.long)
            .unsqueeze(0)
            .to(model.device)
        )

        prompt_tokens = model.tokenizer(prompt_seq, add_special_tokens=False)[
            "input_ids"
        ]
        start_tokens = [47, 63]
        prompt_ids_list = list(start_tokens) + list(prompt_tokens)
        input_ids = torch.tensor(
            prompt_ids_list, dtype=torch.long, device=model.device
        ).unsqueeze(0)

        legacy_residue_scores: list[np.ndarray] = []
        comp_ids_2d = completion_ids.squeeze(0)
        with torch.no_grad():
            for i in range(comp_ids_2d.shape[0]):
                comp = comp_ids_2d[i : i + 1]
                full_ids = torch.cat([input_ids, comp], dim=-1)
                start_ix = input_ids.shape[-1]
                outputs = model.model(input_ids=full_ids, use_cache=False)
                labels = torch.where(
                    full_ids == model.tokenizer.pad_token_id,
                    -100,
                    full_ids.clone(),
                )
                ll = log_likelihood_from_outputs(outputs, labels, start_ix=start_ix)
                ll_np = ll[0].cpu().float().numpy()
                shift_labels = labels[..., start_ix + 1 :]
                mask = (shift_labels[0] != -100).cpu().numpy()
                legacy_residue_scores.append(ll_np[mask])

        with torch.no_grad():
            new_residue_scores = model.score_seqs(
                input_ids,
                completion_ids,
                use_cache=False,
                batch_size=1,
                return_per_residue=True,
            )

        assert isinstance(new_residue_scores, list)
        assert len(new_residue_scores) == len(legacy_residue_scores)
        for legacy, new in zip(legacy_residue_scores, new_residue_scores):
            assert new.shape == legacy.shape, (new.shape, legacy.shape)
            np.testing.assert_allclose(new, legacy, rtol=1e-5, atol=1e-6)

        with torch.no_grad():
            kv_residue_scores = model.score_seqs(
                input_ids,
                completion_ids,
                use_cache=True,
                batch_size=1,
                return_per_residue=True,
            )
        assert len(kv_residue_scores) == len(legacy_residue_scores)
        for legacy, kv in zip(legacy_residue_scores, kv_residue_scores):
            assert kv.shape == legacy.shape
            np.testing.assert_allclose(kv, legacy, rtol=1e-4, atol=1e-5)

    def test_score_per_residue_batched_matches_unbatched(self):
        """Per-residue scoring with batch_size>1 (KV-cache path) must
        produce the same per-position log-likelihoods as the unbatched
        path, for variants of different lengths that therefore share a
        mini-batch with trailing padding.
        """
        pf = self._make_profam()
        sequences = [
            "ACDEFGHIKLMNPQRSTVWY",
            "ACDEFGHIKLMNPQRSTVWYAC",
            "ACDEFGHIKLMNP",
            "ACDEFGHIKLMNPQRSTVWYACDE",
        ]
        prompt = ["ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIKLMNPQRSTVWY"]

        result_unbatched = pf.score(
            sequences=sequences,
            prompt=prompt,
            ensemble_size=1,
            max_tokens=2048,
            scoring_max_tokens=1,  # forces batch_size=1 in per-residue path
            per_residue=True,
            use_diversity_weights=False,
            seed=42,
        )

        result_batched = pf.score(
            sequences=sequences,
            prompt=prompt,
            ensemble_size=1,
            max_tokens=2048,
            scoring_max_tokens=100_000,  # fits everything into one batch
            per_residue=True,
            use_diversity_weights=False,
            seed=42,
        )

        assert result_unbatched.residue_scores is not None
        assert result_batched.residue_scores is not None
        assert len(result_unbatched.residue_scores) == len(sequences)
        assert len(result_batched.residue_scores) == len(sequences)
        for i, (u, b) in enumerate(
            zip(result_unbatched.residue_scores, result_batched.residue_scores)
        ):
            assert u.shape == b.shape, (i, u.shape, b.shape)
            np.testing.assert_allclose(u, b, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("ensemble_size", [1, 3])
    def test_score_per_residue_mean_equals_scores(self, ensemble_size):
        """With per_residue=True, result.scores[i] must equal
        result.residue_scores[i].mean() for every candidate -- proof that
        both outputs come from a single ensemble pass rather than two
        independent runs with different prompts.
        """
        pf = self._make_profam()
        result = pf.score(
            sequences=[
                "ACDEFGHIKLMNPQRSTVWY",
                "ACDEFGHIKLMNPQRSTVWYAC",
                "ACDEFGHIKLMNP",
            ],
            prompt=[
                "ACDEFGHIKLMNPQRSTVWY",
                "WYVTSRQPNMLKIHGFEDCA",
                "ACDEFGHIKLMNPQRSTVWYAC",
                "MCDEFGHIKLMNPQRSTVWY",
            ],
            ensemble_size=ensemble_size,
            max_tokens=2048,
            per_residue=True,
            use_diversity_weights=False,
            seed=13,
        )
        assert result.residue_scores is not None
        assert result.scores.shape == (len(result.sequences),)
        for i, residue_ll in enumerate(result.residue_scores):
            assert residue_ll.size > 0
            np.testing.assert_allclose(
                result.scores[i], float(residue_ll.mean()), rtol=1e-6, atol=1e-6
            )

    @pytest.mark.parametrize("ensemble_size", [1, 3])
    def test_score_per_residue_scores_match_non_per_residue(self, ensemble_size):
        """With the same seed, pf.score(per_residue=True) and
        pf.score(per_residue=False) must return (numerically) the same
        scores: turning per-residue on must not change the scoring work,
        only expose the intermediate per-position log-likelihoods.
        """
        pf = self._make_profam()
        common_kwargs = dict(
            sequences=[
                "ACDEFGHIKLMNPQRSTVWY",
                "ACDEFGHIKLMNPQRSTVWYAC",
            ],
            prompt=[
                "ACDEFGHIKLMNPQRSTVWY",
                "WYVTSRQPNMLKIHGFEDCA",
                "ACDEFGHIKLMNPQRSTVWYAC",
            ],
            ensemble_size=ensemble_size,
            max_tokens=2048,
            use_diversity_weights=False,
            seed=29,
        )
        result_mean_only = pf.score(per_residue=False, **common_kwargs)
        result_per_residue = pf.score(per_residue=True, **common_kwargs)
        np.testing.assert_allclose(
            result_mean_only.scores,
            result_per_residue.scores,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_score_per_residue_no_context(self):
        """No-context per-residue path should also satisfy the
        scores[i] == residue_scores[i].mean() invariant.
        """
        pf = self._make_profam()
        result = pf.score(
            sequences=[
                "ACDEFGHIKLMNPQRSTVWY",
                "ACDEFGHIKLMNPQRSTVWYAC",
            ],
            prompt=None,
            per_residue=True,
            seed=5,
        )
        assert result.residue_scores is not None
        for i, residue_ll in enumerate(result.residue_scores):
            np.testing.assert_allclose(
                result.scores[i], float(residue_ll.mean()), rtol=1e-6, atol=1e-6
            )
