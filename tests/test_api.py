"""Tests for the public ProFam Python API."""

import numpy as np
import pytest
import torch

from profam.api import GenerationResult, ProFam, ScoringResult, _build_protein_document
from profam.data.objects import ProteinDocument


def test_build_protein_document_from_strings():
    seqs = ["ACDEFGHIK", "LMNPQRST"]
    doc = _build_protein_document(seqs)
    assert isinstance(doc, ProteinDocument)
    assert list(doc.sequences) == seqs
    assert len(doc.accessions) == 2


def test_generation_result_structure():
    result = GenerationResult(sequences=["ACDE", "FGHI"], scores=[-1.0, -2.0])
    assert len(result.sequences) == 2
    assert len(result.scores) == 2


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
