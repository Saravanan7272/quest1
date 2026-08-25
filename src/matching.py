"""
src/matching.py
Centralized text normalization and similarity matching algorithms.
"""

import re
from rapidfuzz import fuzz

def normalize(text: str) -> str:
    """
    Robust whitespace and case normalization.
    """
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def compute_asr_score(
    segment_text: str,
    target_text: str,
    partial_weight: float = 0.70,
    coverage_weight: float = 0.30
) -> float:
    """
    Compute ASR candidate match score combining partial phrase ratio and token coverage.
    """
    norm_seg = normalize(segment_text)
    norm_target = normalize(target_text)
    
    if not norm_seg or not norm_target:
        return 0.0
        
    partial_sim = fuzz.partial_ratio(norm_seg, norm_target) / 100.0
    
    seg_tokens = set(norm_seg.split())
    target_tokens = set(norm_target.split())
    
    if not target_tokens:
        coverage = 0.0
    else:
        coverage = len(target_tokens & seg_tokens) / len(target_tokens)
        
    score = (partial_weight * partial_sim) + (coverage_weight * coverage)
    return round(score, 4)

def compute_ocr_score(
    ocr_text: str,
    target_text: str,
    character_weight: float = 0.60,
    token_weight: float = 0.40
) -> float:
    """
    Compute visual OCR box match strength combining character similarity and token coverage.
    """
    norm_ocr = normalize(ocr_text)
    norm_target = normalize(target_text)
    
    if not norm_ocr or not norm_target:
        return 0.0
        
    char_sim = fuzz.ratio(norm_ocr, norm_target) / 100.0
    
    ocr_tokens = set(norm_ocr.split())
    target_tokens = set(norm_target.split())
    
    if not target_tokens:
        coverage = 0.0
    else:
        coverage = len(target_tokens & ocr_tokens) / len(target_tokens)
        
    match_strength = (character_weight * char_sim) + (token_weight * coverage)
    return round(match_strength, 4)

def classify_match_level(
    match_strength: float,
    similarity_threshold: float = 0.75,
    high_match_threshold: float = 0.85,
    has_boxes: bool = True
) -> str:
    """
    Classify result status level into HIGH, MEDIUM, AMBIGUOUS, or NOT_FOUND.
    """
    if not has_boxes or match_strength == 0.0:
        return "NOT_FOUND"
    if match_strength >= high_match_threshold:
        return "HIGH"
    if match_strength >= similarity_threshold:
        return "MEDIUM"
    return "AMBIGUOUS"
