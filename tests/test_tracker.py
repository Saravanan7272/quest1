"""
tests/test_tracker.py
Unit tests for TextTracker: IoU calculation, simultaneous multi-box tracking,
text disappearance/reappearance, and query relevance best frame selection.
"""

from src.models import TrackedBox
from src.text_tracker import TextTracker, compute_iou

def test_compute_iou():
    boxA = [10.0, 10.0, 50.0, 50.0]
    boxB = [10.0, 10.0, 50.0, 50.0]
    assert compute_iou(boxA, boxB) == 1.0

    boxC = [100.0, 100.0, 150.0, 150.0]
    assert compute_iou(boxA, boxC) == 0.0

# TEST 2 — Simultaneous multi-box tracking (Title + Subtitle)
def test_simultaneous_multibox_tracking():
    tracker = TextTracker(iou_threshold=0.5, max_gap_seconds=0.5)

    # Frame 1 at ts=0.0: Title (top) and Subtitle (bottom)
    box_title_f1 = TrackedBox(timestamp=0.0, bbox=[10.0, 10.0, 100.0, 30.0], detection_confidence=0.9)
    box_sub_f1 = TrackedBox(timestamp=0.0, bbox=[10.0, 100.0, 200.0, 130.0], detection_confidence=0.9)

    tracker.update(0.0, [box_title_f1, box_sub_f1])
    assert len(tracker.active_tracks) == 2

    track1 = tracker.active_tracks[0]
    track2 = tracker.active_tracks[1]

    # Frame 2 at ts=0.1: Slightly moved Title and Subtitle
    box_title_f2 = TrackedBox(timestamp=0.1, bbox=[12.0, 11.0, 102.0, 31.0], detection_confidence=0.9)
    box_sub_f2 = TrackedBox(timestamp=0.1, bbox=[11.0, 101.0, 201.0, 131.0], detection_confidence=0.9)

    tracker.update(0.1, [box_title_f2, box_sub_f2])
    assert len(tracker.active_tracks) == 2

    # Verify no track received multiple boxes from the same frame
    assert len(track1.boxes) == 2
    assert len(track2.boxes) == 2
    assert track1.track_id != track2.track_id

# TEST 3 — Text disappears and reappears
def test_text_disappears_and_reappears():
    tracker = TextTracker(iou_threshold=0.5, max_gap_seconds=0.5)

    # Event 1: 0.0s to 0.2s
    b1 = TrackedBox(timestamp=0.0, bbox=[10.0, 10.0, 50.0, 50.0], detection_confidence=0.9)
    b2 = TrackedBox(timestamp=0.2, bbox=[10.0, 10.0, 50.0, 50.0], detection_confidence=0.9)
    tracker.update(0.0, [b1])
    tracker.update(0.2, [b2])

    # Gap of 1.0s (exceeds max_gap_seconds=0.5s)
    # Event 2 at ts=1.2s
    b3 = TrackedBox(timestamp=1.2, bbox=[10.0, 10.0, 50.0, 50.0], detection_confidence=0.9)
    tracker.update(1.2, [b3])

    # Verify Event 1 was aged out into completed_tracks
    all_tracks = tracker.finalize()
    assert len(all_tracks) == 2
    assert all_tracks[0].track_id != all_tracks[1].track_id
    assert all_tracks[0].end == 0.2
    assert all_tracks[1].start == 1.2

def test_update_best_frame_by_query_relevance():
    tracker = TextTracker()
    b = TrackedBox(timestamp=0.0, bbox=[10.0, 10.0, 50.0, 50.0], detection_confidence=0.9)
    tracker.update(0.0, [b])

    track_id = tracker.active_tracks[0].track_id
    tracker.update_best_frame(track_id, timestamp=0.0, text="LOW RELEVANCE", query_relevance=0.40)
    tracker.update_best_frame(track_id, timestamp=0.5, text="HIGH RELEVANCE", query_relevance=0.95)

    track = tracker.active_tracks[0]
    assert track.best_text == "HIGH RELEVANCE"
    assert track.best_frame_timestamp == 0.5
    assert track.best_relevance == 0.95
