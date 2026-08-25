"""
src/candidate_association.py
Associates ASR audio evidence windows with Visual discovery candidates.
"""

import logging
from typing import List, Optional, Dict
from src.models import Candidate, ModalityScores
from src.scoring import fuse_scores

logger = logging.getLogger(__name__)

def associate_audio_visual_evidence(
    asr_candidates: List[Candidate],
    visual_candidates: List[Candidate],
    weights: Dict[str, float],
    association_window: float = 2.0
) -> List[Candidate]:
    """
    Associate ASR audio evidence candidates and Visual discovery candidates
    within an association_window (default 2.0 seconds) into unified multimodal candidates.
    """
    merged: List[Candidate] = []
    used_visual_indices = set()

    for asr_cand in asr_candidates:
        matched_visual: Optional[Candidate] = None
        best_diff = association_window

        for idx, vis_cand in enumerate(visual_candidates):
            diff = abs(asr_cand.timestamp - vis_cand.timestamp)
            if diff <= best_diff:
                best_diff = diff
                matched_visual = vis_cand
                used_visual_indices.add(idx)

        if matched_visual:
            combined_scores = ModalityScores(
                asr=asr_cand.scores.asr,
                ocr=matched_visual.scores.ocr
            )
            fused = fuse_scores(combined_scores, weights)
            merged.append(
                Candidate(
                    timestamp=matched_visual.timestamp,
                    frame_number=matched_visual.frame_number,
                    text=matched_visual.text,
                    scores=combined_scores,
                    bbox=matched_visual.bbox,
                    image_path=matched_visual.image_path,
                    source="multimodal",
                    fused_score=fused
                )
            )
        else:
            fused = fuse_scores(asr_cand.scores, weights)
            asr_cand.fused_score = fused
            merged.append(asr_cand)

    for idx, vis_cand in enumerate(visual_candidates):
        if idx not in used_visual_indices:
            fused = fuse_scores(vis_cand.scores, weights)
            vis_cand.fused_score = fused
            merged.append(vis_cand)

    return merged
