"""
tests/test_scoring.py
Unit tests for ModalityScores fusion and missing-modality normalization.
"""

from src.models import ModalityScores
from src.scoring import fuse_scores, determine_match_level

def test_fuse_scores_all_modalities():
    scores = ModalityScores(asr=1.0, ocr=1.0, semantic=1.0)
    weights = {"asr": 0.30, "ocr": 0.50, "semantic": 0.20}
    assert fuse_scores(scores, weights) == 1.0

def test_fuse_scores_missing_modality_normalization():
    # Only ASR available (0.90)
    scores_asr_only = ModalityScores(asr=0.90, ocr=None, semantic=None)
    weights = {"asr": 0.30, "ocr": 0.50, "semantic": 0.20}
    # Normalized: 0.90 * (0.30 / 0.30) = 0.90
    assert fuse_scores(scores_asr_only, weights) == 0.90

    # OCR and ASR available
    scores_dual = ModalityScores(asr=0.80, ocr=1.00, semantic=None)
    # Available weights: 0.30 + 0.50 = 0.80
    # Fused: 0.80 * (0.30/0.80) + 1.00 * (0.50/0.80) = 0.30 + 0.625 = 0.925 -> round 0.925
    assert fuse_scores(scores_dual, weights) == 0.925

def test_determine_match_level():
    assert determine_match_level(0.92) == "HIGH"
    assert determine_match_level(0.78) == "MEDIUM"
    assert determine_match_level(0.50) == "BELOW_THRESHOLD"
    assert determine_match_level(0.0) == "NOT_FOUND"
