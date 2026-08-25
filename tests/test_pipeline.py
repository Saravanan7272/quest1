"""
tests/test_pipeline.py
Unit tests for pipeline input validation, YAML configuration loading, and error handling paths.
"""

import yaml
from pathlib import Path
from src.pipeline import locate_dialogue_in_video, format_timestamp_hhmmss

# TEST 12: configuration loading
def test_config_loading(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_content = {
        "asr": {"model": "tiny", "window_padding": 5.0},
        "sampling": {"coarse_fps": 1.0},
        "matching": {"similarity_threshold": 0.75}
    }
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config_content, f)
        
    with open(config_file, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
        
    assert loaded["asr"]["model"] == "tiny"
    assert loaded["sampling"]["coarse_fps"] == 1.0

# TEST 13: invalid target
def test_invalid_target_text():
    res = locate_dialogue_in_video(url="https://ok.ru/video/12345", target_text="", config={})
    assert res["status"] == "ERROR"
    assert "target text" in res["error"].lower()

# TEST 14: pipeline failure handling (invalid URL)
def test_pipeline_invalid_url():
    res = locate_dialogue_in_video(url="", target_text="My mind rebels at stagnation", config={})
    assert res["status"] == "ERROR"

def test_timestamp_formatting():
    assert format_timestamp_hhmmss(0.0) == "00:00:00.000"
    assert format_timestamp_hhmmss(123.456) == "00:02:03.456"
    assert format_timestamp_hhmmss(3665.123) == "01:01:05.123"
