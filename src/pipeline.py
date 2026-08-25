"""
src/pipeline.py
Main pipeline orchestrator combining ASR Primary Path, Visual Discovery Path,
Candidate Association, Missing-Modality Score Fusion, Temporal Deduplication, and Top-K output.
"""

import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.models import Candidate, ModalityScores, SearchStats
from src.acquisition import download_video, AcquisitionError
from src.asr import search_audio_for_target
from src.sampling import generate_sample_timestamps, extract_frames_at_timestamps
from src.ocr import OCREngineAdapter
from src.matching import compute_ocr_score
from src.visual_pipeline import run_visual_discovery
from src.candidate_association import associate_audio_visual_evidence
from src.scoring import fuse_scores, determine_match_level

logger = logging.getLogger(__name__)

def format_timestamp_hhmmss(seconds: float) -> str:
    """Format float seconds as HH:MM:SS.sss."""
    millis = int(round((seconds - int(seconds)) * 1000))
    total_sec = int(seconds)
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    secs = total_sec % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

def associate_and_fuse_candidates(
    asr_candidates: List[Candidate],
    visual_candidates: List[Candidate],
    weights: Dict[str, float],
    association_window: float = 2.0
) -> List[Candidate]:
    """
    Merge ASR and Visual candidates within association_window into multimodal candidates
    and apply score fusion with missing-modality normalization.
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
            # Combined multimodal candidate
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
            # ASR only candidate
            fused = fuse_scores(asr_cand.scores, weights)
            asr_cand.fused_score = fused
            merged.append(asr_cand)

    # Append remaining visual-only candidates
    for idx, vis_cand in enumerate(visual_candidates):
        if idx not in used_visual_indices:
            fused = fuse_scores(vis_cand.scores, weights)
            vis_cand.fused_score = fused
            merged.append(vis_cand)

    return merged

def deduplicate_top_k(
    candidates: List[Candidate],
    top_k: int = 3,
    dedup_window: float = 2.0
) -> List[Candidate]:
    """
    Sort candidates by fused_score descending and deduplicate occurrences within dedup_window.
    """
    sorted_cands = sorted(candidates, key=lambda c: (c.fused_score or 0.0), reverse=True)
    deduped: List[Candidate] = []

    for cand in sorted_cands:
        is_dup = False
        for prev in deduped:
            if abs(cand.timestamp - prev.timestamp) < dedup_window:
                is_dup = True
                break
        if not is_dup:
            deduped.append(cand)
        if len(deduped) >= top_k:
            break

    return deduped

def locate_dialogue_in_video(
    url: str,
    target_text: str,
    config: Dict[str, Any],
    output_dir: Path = Path("outputs"),
    temp_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Main Orchestrator for Dual-Path Video Dialogue Locator.
    """
    start_wall = time.time()
    
    if not url or not url.strip():
        return {"status": "ERROR", "error": "Invalid or empty video URL provided."}
    if not target_text or not target_text.strip():
        return {"status": "ERROR", "error": "Invalid or empty target text provided."}

    asr_cfg = config.get("asr", {})
    sampling_cfg = config.get("sampling", {})
    matching_cfg = config.get("matching", {})
    scoring_cfg = config.get("scoring", {})
    output_cfg = config.get("output", {})
    weights = scoring_cfg.get("weights", {"asr": 0.30, "ocr": 0.50, "semantic": 0.20})

    output_dir.mkdir(parents=True, exist_ok=True)
    
    cleanup_temp = False
    if temp_dir is None:
        temp_workspace = Path(tempfile.mkdtemp(prefix="loc_vdir_"))
        cleanup_temp = True
    else:
        temp_workspace = temp_dir
        temp_workspace.mkdir(parents=True, exist_ok=True)

    search_stats = SearchStats()

    try:
        # Stage 1: Video Acquisition
        logger.info("--- STAGE 1: ACQUISITION ---")
        acq_dir = temp_workspace / "acquisition"
        video_meta = download_video(url=url, output_dir=acq_dir)
        
        # Stage 2: ASR Primary Path
        logger.info("--- STAGE 2: ASR PRIMARY PATH ---")
        asr_res = search_audio_for_target(
            audio_video_path=video_meta.video_path,
            target_text=target_text,
            video_duration=video_meta.duration,
            model_name=asr_cfg.get("model", "base"),
            device=asr_cfg.get("device", "cpu"),
            compute_type=asr_cfg.get("compute_type", "int8"),
            language=asr_cfg.get("language"),
            window_padding=asr_cfg.get("window_padding", 5.0),
            candidate_threshold=asr_cfg.get("candidate_threshold", 0.60),
            partial_weight=asr_cfg.get("partial_weight", 0.70),
            coverage_weight=asr_cfg.get("coverage_weight", 0.30)
        )

        asr_candidates: List[Candidate] = []
        if asr_res.candidate_found:
            # Sample frames in ASR candidate window & verify with OCR
            coarse_fps = sampling_cfg.get("coarse_fps", 1.0)
            sample_ts = generate_sample_timestamps(asr_res.start_time, asr_res.end_time, coarse_fps=coarse_fps)
            asr_frames_dir = temp_workspace / "asr_frames"
            frame_records = extract_frames_at_timestamps(video_meta.video_path, sample_ts, asr_frames_dir)
            
            ocr_engine = OCREngineAdapter(lang=config.get("ocr", {}).get("lang", "en"))
            found_ocr_match = False
            
            for rec in frame_records:
                search_stats.ocr_calls += 1
                ocr_boxes = ocr_engine.run_ocr(rec.path)
                for box in ocr_boxes:
                    score = compute_ocr_score(
                        box.text, target_text,
                        character_weight=matching_cfg.get("character_weight", 0.60),
                        token_weight=matching_cfg.get("token_weight", 0.40)
                    )
                    if score >= matching_cfg.get("similarity_threshold", 0.75):
                        found_ocr_match = True
                        frame_num = int(round(rec.timestamp * video_meta.fps))
                        c_scores = ModalityScores(asr=asr_res.best_score, ocr=score)
                        asr_candidates.append(
                            Candidate(
                                timestamp=round(rec.timestamp, 3),
                                frame_number=frame_num,
                                text=box.text,
                                scores=c_scores,
                                bbox=box.bounding_rect,
                                image_path=str(rec.path),
                                source="multimodal"
                            )
                        )
            
            # If no OCR match on screen, preserve spoken audio candidate!
            if not found_ocr_match and frame_records:
                best_rec = frame_records[0]
                frame_num = int(round(best_rec.timestamp * video_meta.fps))
                asr_candidates.append(
                    Candidate(
                        timestamp=round(asr_res.start_time, 3),
                        frame_number=frame_num,
                        text=target_text,
                        scores=ModalityScores(asr=asr_res.best_score),
                        image_path=str(best_rec.path),
                        source="asr"
                    )
                )

        # Stage 3: Visual Discovery Path
        logger.info("--- STAGE 3: VISUAL DISCOVERY PATH ---")
        visual_candidates = []
        if config.get("visual_scout", {}).get("enabled", True):
            visual_candidates, vis_stats = run_visual_discovery(
                video_path=video_meta.video_path,
                target_text=target_text,
                duration=video_meta.duration,
                fps=video_meta.fps,
                config=config,
                temp_dir=temp_workspace
            )
            search_stats.scout_frames += vis_stats.scout_frames
            search_stats.detector_frames += vis_stats.detector_frames
            search_stats.ocr_calls += vis_stats.ocr_calls
            search_stats.tracked_events += vis_stats.tracked_events

        # Stage 4: Candidate Association & Score Fusion
        logger.info("--- STAGE 4: ASSOCIATION & SCORE FUSION ---")
        all_candidates = associate_and_fuse_candidates(
            asr_candidates, visual_candidates, weights,
            association_window=config.get("visual_scout", {}).get("trigger_merge_window", 2.0)
        )

        # Filter candidates above similarity threshold
        sim_thresh = matching_cfg.get("similarity_threshold", 0.75)
        valid_candidates = [c for c in all_candidates if (c.fused_score or 0.0) >= sim_thresh]

        # Deduplicate & Select Top-K
        top_k = output_cfg.get("top_k", 3)
        top_candidates = deduplicate_top_k(valid_candidates, top_k=top_k, dedup_window=output_cfg.get("dedup_window", 2.0))

        search_stats.candidates_found = len(valid_candidates)
        search_stats.runtime_seconds = round(time.time() - start_wall, 2)

        # Format output contract
        formatted_results = []
        for cand in top_candidates:
            # Copy evidence image
            evidence_filename = f"evidence_{cand.timestamp:.3f}s.jpg"
            evidence_dest = output_dir / evidence_filename
            if cand.image_path and Path(cand.image_path).exists():
                shutil.copy2(cand.image_path, evidence_dest)
                img_rel_path = str(evidence_dest).replace("\\", "/")
            else:
                img_rel_path = ""

            match_level = determine_match_level(cand.fused_score or 0.0, sim_thresh, matching_cfg.get("high_match_threshold", 0.85))

            sources = []
            if cand.scores.asr is not None:
                sources.append("asr")
            if cand.scores.ocr is not None:
                sources.append("ocr")

            res_entry = {
                "timestamp": format_timestamp_hhmmss(cand.timestamp),
                "timestamp_seconds": cand.timestamp,
                "frame_number": cand.frame_number,
                "extracted_text": cand.text,
                "match_strength": cand.fused_score or 0.0,
                "match_level": match_level,
                "image_path": img_rel_path,
                "sources": sources,
                "evidence": {
                    "speech_match": (cand.scores.asr is not None and cand.scores.asr >= asr_cfg.get("candidate_threshold", 0.50)),
                    "visual_text_match": (cand.scores.ocr is not None and cand.scores.ocr >= sim_thresh)
                },
                "scores": {
                    "asr": cand.scores.asr,
                    "ocr": cand.scores.ocr,
                    "semantic": cand.scores.semantic
                }
            }
            formatted_results.append(res_entry)

        status_str = "FOUND" if formatted_results else "NOT_FOUND"

        return {
            "status": status_str,
            "results": formatted_results,
            "total_candidates": len(formatted_results),
            "search_summary": {
                "scout_frames": search_stats.scout_frames,
                "detector_frames": search_stats.detector_frames,
                "ocr_calls": search_stats.ocr_calls,
                "candidates_found": search_stats.candidates_found,
                "tracked_events": search_stats.tracked_events,
                "runtime_seconds": search_stats.runtime_seconds
            }
        }

    except AcquisitionError as ae:
        logger.error(f"Acquisition stage error: {ae}")
        return {"status": "ERROR", "error": str(ae)}
    except Exception as e:
        logger.error(f"Pipeline execution error: {e}", exc_info=True)
        return {"status": "ERROR", "error": str(e)}
    finally:
        if cleanup_temp and temp_workspace.exists():
            try:
                shutil.rmtree(temp_workspace)
            except Exception:
                pass
