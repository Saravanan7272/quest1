"""
tests/test_matching.py
Unit tests for text normalization, ASR phrase/token scoring, and OCR box similarity.
"""

import pytest
from src.matching import (
    normalize,
    compute_asr_score,
    compute_ocr_score,
    classify_match_level
)

# TEST 1: normalize()
def test_normalize():
    assert normalize("  My   MIND rebels   at STAGNATION ") == "my mind rebels at stagnation"
    assert normalize("") == ""
    assert normalize("hello\n\nworld\ttest") == "hello world test"

# TEST 2: exact ASR match
def test_exact_asr_match():
    score = compute_asr_score("My mind rebels at stagnation", "My mind rebels at stagnation")
    assert score == 1.0

# TEST 3: partial ASR match
def test_partial_asr_match():
    score = compute_asr_score("Well my mind rebels at stagnation indeed", "My mind rebels at stagnation")
    assert score > 0.85

# TEST 4: weak common-word match
def test_weak_common_word_match():
    # Only single common word 'the' should produce low score
    score = compute_asr_score("the cat sat on the mat", "My mind rebels at the stagnation")
    assert score < 0.50

# TEST 5: token coverage
def test_token_coverage():
    score = compute_asr_score("mind rebels stagnation", "my mind rebels at stagnation")
    # 3 out of 4 target tokens matched
    assert score >= 0.70

# TEST 6: visual character similarity
def test_visual_character_similarity():
    score = compute_ocr_score("My mind rebels at stagnatn", "My mind rebels at stagnation")
    assert score >= 0.85

# TEST 7: visual token coverage
def test_visual_token_coverage():
    score = compute_ocr_score("mind rebels at stagnation", "my mind rebels at stagnation")
    assert score >= 0.80

def test_classify_match_level():
    assert classify_match_level(0.92) == "HIGH"
    assert classify_match_level(0.78) == "MEDIUM"
    assert classify_match_level(0.50) == "AMBIGUOUS"
    assert classify_match_level(0.0, has_boxes=False) == "NOT_FOUND"
