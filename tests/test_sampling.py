"""
tests/test_sampling.py
Unit tests for sampling timestamp array generation.
"""

import pytest
import numpy as np
from src.sampling import generate_sample_timestamps

# TEST 11: sampling timestamp generation
def test_generate_sample_timestamps():
    timestamps = generate_sample_timestamps(10.0, 15.0, coarse_fps=1.0)
    assert isinstance(timestamps, np.ndarray)
    assert len(timestamps) == 6
    assert np.isclose(timestamps[0], 10.0)
    assert np.isclose(timestamps[-1], 15.0)

def test_generate_sample_timestamps_zero_range():
    timestamps = generate_sample_timestamps(5.0, 5.0, coarse_fps=1.0)
    assert len(timestamps) == 1
    assert timestamps[0] == 5.0
