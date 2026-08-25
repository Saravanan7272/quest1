"""
tests/test_visual_scout.py
Unit tests for visual change detection, periodic detector insurance, and trigger merging.
"""

from src.visual_scout import generate_periodic_triggers, merge_triggers

# Test 1 — Visual text with no scene change (Periodic Detector protection)
def test_periodic_detector_triggers():
    # 10 second video duration with 0.5 FPS periodic detector (step = 2.0s)
    triggers = generate_periodic_triggers(duration=10.0, periodic_fps=0.5)
    assert len(triggers) == 6
    assert triggers[0] == 0.0
    assert triggers[1] == 2.0
    assert triggers[-1] == 10.0

def test_merge_triggers_deduplication():
    scout_triggers = [1.0, 5.0, 9.0]
    periodic_triggers = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    merged = merge_triggers(scout_triggers, periodic_triggers, merge_window=2.0)
    
    # 0.0 and 1.0 merged -> 0.0; 2.0 ignored (within 2s); 4.0 & 5.0 merged -> 4.0, etc.
    assert len(merged) > 0
    assert merged[0] == 0.0
    assert all((merged[i+1] - merged[i]) >= 2.0 for i in range(len(merged) - 1))
