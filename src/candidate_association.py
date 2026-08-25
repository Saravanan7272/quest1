"""
src/candidate_association.py
Associates ASR audio evidence windows with Visual discovery candidates.
"""

import logging
from typing import List, Optional, Dict
from src.models import Candidate, ModalityScores, EvidenceMetadata
from src.scoring import fuse_scores

logger = logging.getLogger(__name__)

def associate_audio_visual_evidence(
    asr_candidates: List[Candidate],
    visual_candidates: List[Candidate],
    weights: Dict[str, float],
    association_window: float = 5.0,
    ocr_min_threshold: float = 0.45
) -> List[Candidate]:
    """
    Associate ASR audio candidates and Visual discovery candidates using flexible
    temporal proximity and multi-factor OCR evaluation.
    """
    merged: List[Candidate] = []
    used_visual_indices = set()

    for asr_cand in asr_candidates:
        matched_visual: Optional[Candidate] = None
        best_diff = association_window

        # ASR Query bounds
        q_start = asr_cand.asr_span.query_start if asr_cand.asr_span else asr_cand.timestamp
        q_end = asr_cand.asr_span.query_end if asr_cand.asr_span else asr_cand.timestamp

        for idx, vis_cand in enumerate(visual_candidates):
            v_ts = vis_cand.timestamp
            v_start = vis_cand.visual_span.start if vis_cand.visual_span else v_ts
            v_end = vis_cand.visual_span.end if vis_cand.visual_span else v_ts

            # Check temporal compatibility
            is_compat = (v_start <= q_end + association_window) and (v_end >= q_start - association_window)
            diff = min(abs(v_ts - q_start), abs(v_ts - q_end), abs(v_start - q_start))

            ocr_score = vis_cand.scores.ocr or 0.0

            if is_compat and ocr_score >= ocr_min_threshold and diff <= best_diff:
                best_diff = diff
                matched_visual = vis_cand
                used_visual_indices.add(idx)

        if matched_visual:
            combined_scores = ModalityScores(
                asr=asr_cand.scores.asr,
                ocr=matched_visual.scores.ocr
            )
            fused = fuse_scores(combined_scores, weights)
            
            # Evidence frame comes from visual candidate!
            ev_meta = EvidenceMetadata(
                timestamp_seconds=matched_visual.timestamp,
                frame_number=matched_visual.frame_number,
                image_path=matched_visual.image_path or "",
                source=["asr", "ocr"]
            )
            
            cand = Candidate(
                timestamp=matched_visual.timestamp,
                frame_number=matched_visual.frame_number,
                text=matched_visual.text,
                scores=combined_scores,
                bbox=matched_visual.bbox,
                image_path=matched_visual.image_path,
                source="multimodal",
                fused_score=fused,
                speech_match=True,
                visual_text_match=True,
                sources=["asr", "ocr"],
                asr_span=asr_cand.asr_span,
                visual_span=matched_visual.visual_span,
                evidence=ev_meta
            )
            merged.append(cand)
            logger.info(
                f"Multi-metric Association: ASR span [{q_start:.2f}s, {q_end:.2f}s] + "
                f"Visual track [{matched_visual.timestamp:.2f}s, text='{matched_visual.text}'] -> MULTIMODAL MATCH (fused={fused:.4f})"
            )
        else:
            fused = fuse_scores(asr_cand.scores, weights)
            asr_cand.fused_score = fused
            asr_cand.speech_match = True
            asr_cand.visual_text_match = False
            asr_cand.sources = ["asr"]
            asr_cand.evidence = EvidenceMetadata(
                timestamp_seconds=asr_cand.timestamp,
                frame_number=asr_cand.frame_number,
                image_path=asr_cand.image_path or "",
                source=["asr"]
            )
            merged.append(asr_cand)

    for idx, vis_cand in enumerate(visual_candidates):
        if idx not in used_visual_indices:
            fused = fuse_scores(vis_cand.scores, weights)
            vis_cand.fused_score = fused
            vis_cand.speech_match = False
            vis_cand.visual_text_match = True
            vis_cand.sources = ["ocr"]
            vis_cand.evidence = EvidenceMetadata(
                timestamp_seconds=vis_cand.timestamp,
                frame_number=vis_cand.frame_number,
                image_path=vis_cand.image_path or "",
                source=["ocr"]
            )
            merged.append(vis_cand)

    return merged

def associate_and_fuse_candidates(
    asr_candidates: List[Candidate],
    visual_candidates: List[Candidate],
    weights: Dict[str, float],
    association_window: float = 5.0,
    ocr_min_threshold: float = 0.45
) -> List[Candidate]:
    """
    Wrapper function delegating to associate_audio_visual_evidence.
    """
    return associate_audio_visual_evidence(
        asr_candidates=asr_candidates,
        visual_candidates=visual_candidates,
        weights=weights,
        association_window=association_window,
        ocr_min_threshold=ocr_min_threshold
    )
