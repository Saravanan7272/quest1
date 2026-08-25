"""
tests/test_candidate_association.py
Unit tests for audio/visual candidate association within window.
"""

from src.models import Candidate, ModalityScores, ASRQuerySpan, VisualTrackSpan
from src.candidate_association import associate_audio_visual_evidence, associate_and_fuse_candidates

def test_associate_audio_visual_evidence():
    asr_cand = Candidate(
        timestamp=10.0,
        frame_number=300,
        text="My mind rebels at stagnation",
        scores=ModalityScores(asr=0.90),
        source="asr"
    )

    vis_cand = Candidate(
        timestamp=11.0,
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

def test_associate_and_fuse_candidates_signature():
    asr_cand = Candidate(
        timestamp=10.0,
        frame_number=300,
        text="test",
        scores=ModalityScores(asr=0.90),
        source="asr"
    )
    weights = {"asr": 0.30, "ocr": 0.50, "semantic": 0.20}
    merged = associate_and_fuse_candidates([asr_cand], [], weights, association_window=5.0, ocr_min_threshold=0.45)
    assert len(merged) == 1

def test_multimodal_evidence_timestamp_selected_from_visual_track():
    asr_span = ASRQuerySpan(
        segment_text="You could die. At least tell me your name.",
        segment_start=7.90,
        segment_end=20.68,
        query_start=9.68,
        query_end=11.50,
        score=0.95
    )
    asr_cand = Candidate(
        timestamp=9.68,
        frame_number=290,
        text="At least tell me your name",
        scores=ModalityScores(asr=0.95),
        source="asr",
        asr_span=asr_span
    )
    vis_span = VisualTrackSpan(
        track_id=8,
        start=13.40,
        end=19.10,
        best_frame_timestamp=14.80,
        ocr_text="LLMEYOUR NAME",
        ocr_confidence=0.826,
        query_similarity=0.467
    )
    vis_cand = Candidate(
        timestamp=14.80,
        frame_number=444,
        text="LLMEYOUR NAME",
        scores=ModalityScores(ocr=0.467),
        source="visual",
        visual_span=vis_span
    )

    weights = {"asr": 0.30, "ocr": 0.50, "semantic": 0.20}
    merged = associate_audio_visual_evidence([asr_cand], [vis_cand], weights, association_window=5.0, ocr_min_threshold=0.40)

    assert len(merged) == 1
    m = merged[0]
    assert m.sources == ["asr", "ocr"]
    assert m.speech_match is True
    assert m.visual_text_match is True
    assert m.evidence.timestamp_seconds == 14.80
