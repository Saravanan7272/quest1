"""
tests/test_ocr_adapter.py
Unit tests for OCRBox data structure, spatial reading order sorting, and adapter response parsing.
"""

from pathlib import Path
from src.ocr import OCRBox, sort_ocr_boxes_spatial, OCREngineAdapter

# TEST 8: OCR adapter empty result on non-existent file
def test_ocr_adapter_empty_result(tmp_path):
    adapter = OCREngineAdapter()
    non_existent = tmp_path / "non_existent_frame.jpg"
    boxes = adapter.run_ocr(non_existent)
    assert boxes == []

# TEST 9: OCR adapter bounding rect calculation
def test_ocr_box_bounding_rect():
    polygon = [[10.0, 20.0], [50.0, 20.0], [50.0, 40.0], [10.0, 40.0]]
    box = OCRBox(text="Test Box", confidence=0.95, bbox=polygon)
    x_min, y_min, x_max, y_max = box.bounding_rect
    assert x_min == 10.0
    assert y_min == 20.0
    assert x_max == 50.0
    assert y_max == 40.0

# TEST 10: multiple OCR boxes spatial sorting (reading order: top-to-bottom, left-to-right)
def test_sort_ocr_boxes_spatial():
    box_bottom = OCRBox(text="Bottom Text", confidence=0.9, bbox=[[10.0, 100.0], [50.0, 100.0], [50.0, 120.0], [10.0, 120.0]])
    box_top_left = OCRBox(text="Top Left", confidence=0.9, bbox=[[10.0, 10.0], [50.0, 10.0], [50.0, 30.0], [10.0, 30.0]])
    box_top_right = OCRBox(text="Top Right", confidence=0.9, bbox=[[100.0, 10.0], [150.0, 10.0], [150.0, 30.0], [100.0, 30.0]])
    
    sorted_boxes = sort_ocr_boxes_spatial([box_bottom, box_top_right, box_top_left])
    assert sorted_boxes[0].text == "Top Left"
    assert sorted_boxes[1].text == "Top Right"
    assert sorted_boxes[2].text == "Bottom Text"
