"""
tests/test_candidate_association.py
Unit tests for audio/visual candidate association within window.
"""

from src.models import Candidate, ModalityScores
from src.candidate_association import associate_audio_visual_evidence

def test_associate_audio_visual_evidence():
    asr_cand = Candidate(
        timestamp=10.0,
        frame_number=300,
        text="My mind rebels at stagnation",
        scores=ModalityScores(asr=0.90),
        source="asr"
    )

    vis_cand = Candidate(
        timestamp=11.0,  # Within 2.0s association window
        frame_number=330,
        text="My mind rebels at stagnation",
        scores=ModalityScores(ocr=0.95),
        source="visual"
    )

    weights = {"asr": 0.30, "ocr": 0.50, "semantic": 0.20}
    merged = associate_audio_visual_evidence([asr_cand], [vis_cand], weights, association_window=2.0)

    assert len(merged) == 1
    multimodal = merged[0]
    assert multimodal.source == "multimodal"
    assert multimodal.scores.asr == 0.90
    assert multimodal.scores.ocr == 0.95
    assert multimodal.fused_score > 0.90
