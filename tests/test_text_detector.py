"""
tests/test_text_detector.py
Unit tests for TextDetector module.
"""

from pathlib import Path
from src.text_detector import TextDetector

def test_text_detector_non_existent_file(tmp_path):
    detector = TextDetector()
    non_existent = tmp_path / "non_existent.jpg"
    boxes = detector.detect(non_existent, timestamp=1.0)
    assert boxes == []
