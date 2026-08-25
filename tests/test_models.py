"""
tests/test_models.py
Unit tests for ModalityScores, TrackedBox, TextEvent, Candidate, and SearchStats.
"""

from src.models import ModalityScores, TrackedBox, TextEvent, Candidate, SearchStats

def test_modality_scores_initialization():
    scores = ModalityScores(asr=0.8, ocr=0.9)
    assert scores.asr == 0.8
    assert scores.ocr == 0.9
    assert scores.semantic is None

def test_tracked_box_initialization():
    box = TrackedBox(timestamp=1.5, bbox=[10.0, 10.0, 50.0, 50.0], detection_confidence=0.95)
    assert box.timestamp == 1.5
    assert box.bbox == [10.0, 10.0, 50.0, 50.0]
    assert box.detection_confidence == 0.95
    assert box.ocr_text is None

def test_text_event_initialization():
    event = TextEvent(track_id=1, start=1.0, end=3.0)
    assert event.track_id == 1
    assert event.start == 1.0
    assert event.end == 3.0
    assert event.boxes == []

def test_candidate_initialization():
    cand = Candidate(
        timestamp=2.5,
        frame_number=75,
        text="My mind rebels at stagnation",
        scores=ModalityScores(ocr=0.92),
        source="visual",
        fused_score=0.92
    )
    assert cand.timestamp == 2.5
    assert cand.frame_number == 75
    assert cand.source == "visual"
    assert cand.fused_score == 0.92
