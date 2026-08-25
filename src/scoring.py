"""
src/scoring.py
Score fusion engine supporting separate modality scores and missing-modality normalization.
"""

from typing import Dict, Any, Optional
from src.models import ModalityScores

DEFAULT_WEIGHTS = {
    "asr": 0.30,
    "ocr": 0.50,
    "semantic": 0.20
}

def fuse_scores(
    scores: ModalityScores,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Fuse ASR, OCR, and Semantic scores using missing-modality normalization.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    available = []
    total_weight = 0.0

    if scores.asr is not None:
        available.append((scores.asr, weights.get("asr", 0.30)))
        total_weight += weights.get("asr", 0.30)
        
    if scores.ocr is not None:
        available.append((scores.ocr, weights.get("ocr", 0.50)))
        total_weight += weights.get("ocr", 0.50)

    if scores.semantic is not None:
        available.append((scores.semantic, weights.get("semantic", 0.20)))
        total_weight += weights.get("semantic", 0.20)

    if not available or total_weight <= 0.0:
        return 0.0

    fused = sum(score * (weight / total_weight) for score, weight in available)
    return round(fused, 4)

def determine_match_level(
    fused_score: float,
    similarity_threshold: float = 0.75,
    high_match_threshold: float = 0.85
) -> str:
    """
    Determine discrete match level string.
    """
    if fused_score >= high_match_threshold:
        return "HIGH"
    if fused_score >= similarity_threshold:
        return "MEDIUM"
    if fused_score > 0.0:
        return "BELOW_THRESHOLD"
    return "NOT_FOUND"
