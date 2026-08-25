"""
tests/test_asr_word_span.py
Unit tests for ASR word-level query span extraction (find_query_span).
"""

from dataclasses import dataclass
from src.asr import find_query_span

@dataclass
class MockWord:
    word: str
    start: float
    end: float

def test_find_query_span_exact_match():
    words = [
        MockWord("You", 7.90, 8.20),
        MockWord("could", 8.20, 8.50),
        MockWord("die.", 8.50, 8.90),
        MockWord("At", 9.68, 9.90),
        MockWord("least", 9.90, 10.20),
        MockWord("tell", 10.20, 10.50),
        MockWord("me", 10.50, 10.80),
        MockWord("your", 10.80, 11.10),
        MockWord("name.", 11.10, 11.50)
    ]
    query = "At least tell me your name"
    q_start, q_end = find_query_span(words, query)

    assert abs(q_start - 9.68) < 0.01
    assert abs(q_end - 11.50) < 0.01

def test_find_query_span_empty_words():
    q_start, q_end = find_query_span([], "test query")
    assert q_start == 0.0
    assert q_end == 0.0
