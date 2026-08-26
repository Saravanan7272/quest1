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

from src.models import Candidate, ModalityScores, SearchStats, ASRQuerySpan, VisualTrackSpan, EvidenceMetadata
from src.acquisition import download_video, AcquisitionError
from src.asr import search_audio_for_target
from src.sampling import generate_sample_timestamps, extract_frames_at_timestamps
from src.ocr import OCREngineAdapter
from src.matching import compute_ocr_score
from src.visual_pipeline import run_visual_discovery
from src.candidate_association import associate_audio_visual_evidence, associate_and_fuse_candidates
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

    # Suppress third-party verbose loggers
    for lib in ["httpx", "httpcore", "urllib3", "onnxruntime", "PIL", "matplotlib", "ppocr", "yt_dlp"]:
        logging.getLogger(lib).setLevel(logging.WARNING)

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
        asr_candidates: List[Candidate] = []
        target_search_window: Optional[Tuple[float, float]] = None

        if video_meta.has_audio:
            asr_res = search_audio_for_target(
                audio_video_path=video_meta.video_path,
                target_text=target_text,
                video_duration=video_meta.duration,
                model_name=asr_cfg.get("model", "base"),
                device=asr_cfg.get("device", "cpu"),
                compute_type=asr_cfg.get("compute_type", "int8"),
                language=asr_cfg.get("language"),
                window_padding=asr_cfg.get("window_padding", 5.0),
                candidate_threshold=asr_cfg.get("candidate_threshold", 0.50),
                partial_weight=asr_cfg.get("partial_weight", 0.70),
                coverage_weight=asr_cfg.get("coverage_weight", 0.30)
            )

            if asr_res and asr_res.candidate_found:
                q_start = asr_res.asr_query_start
                q_end = asr_res.asr_query_end
                target_search_window = (max(0.0, q_start - 1.5), min(video_meta.duration, q_end + 3.5))
                
                # Sample frame at query start for ASR evidence fallback
                sample_ts = generate_sample_timestamps(q_start, min(video_meta.duration, q_start + 1.0), coarse_fps=1.0)
                asr_frames_dir = temp_workspace / "asr_frames"
                frame_recs = extract_frames_at_timestamps(video_meta.video_path, sample_ts, asr_frames_dir)
                best_rec_path = str(frame_recs[0].path) if frame_recs else ""
                frame_num = int(round(q_start * video_meta.fps))

                a_span = ASRQuerySpan(
                    segment_text=asr_res.best_segment_text,
                    segment_start=asr_res.asr_segment_start,
                    segment_end=asr_res.asr_segment_end,
                    query_start=q_start,
                    query_end=q_end,
                    score=asr_res.best_score
                )

                asr_cand = Candidate(
                    timestamp=round(q_start, 3),
                    frame_number=frame_num,
                    text=target_text,
                    scores=ModalityScores(asr=asr_res.best_score),
                    image_path=best_rec_path,
                    source="asr",
                    speech_match=True,
                    visual_text_match=False,
                    sources=["asr"],
                    asr_span=a_span,
                    evidence=EvidenceMetadata(
                        timestamp_seconds=round(q_start, 3),
                        frame_number=frame_num,
                        image_path=best_rec_path,
                        source=["asr"]
                    )
                )
                asr_candidates.append(asr_cand)
        else:
            logger.warning("Skipping ASR because downloaded media has no audio stream.")

        # Stage 3: Visual Discovery Path (ASR-guided targeted or global fallback)
        logger.info("--- STAGE 3: VISUAL DISCOVERY PATH ---")
        visual_candidates = []
        if config.get("visual_scout", {}).get("enabled", True):
            visual_candidates, vis_stats = run_visual_discovery(
                video_path=video_meta.video_path,
                target_text=target_text,
                duration=video_meta.duration,
                fps=video_meta.fps,
                config=config,
                temp_dir=temp_workspace,
                target_search_window=target_search_window
            )
            search_stats.scout_frames += vis_stats.scout_frames
            search_stats.detector_frames += vis_stats.detector_frames
            search_stats.ocr_calls += vis_stats.ocr_calls
            search_stats.tracked_events += vis_stats.tracked_events

        # Stage 4: Multi-Metric Association & Score Fusion
        logger.info("--- STAGE 4: ASSOCIATION & SCORE FUSION ---")
        ocr_min_thresh = matching_cfg.get("ocr_min_threshold", 0.45)
        all_candidates = associate_and_fuse_candidates(
            asr_candidates, visual_candidates, weights,
            association_window=config.get("visual_scout", {}).get("trigger_merge_window", 5.0),
            ocr_min_threshold=ocr_min_thresh
        )

        # Filter valid candidates
        sim_thresh = matching_cfg.get("similarity_threshold", 0.60)
        valid_candidates = [c for c in all_candidates if (c.fused_score or 0.0) >= sim_thresh or c.speech_match or c.visual_text_match]

        # Deduplicate & Select Top-K
        top_k = output_cfg.get("top_k", 3)
        top_candidates = deduplicate_top_k(valid_candidates, top_k=top_k, dedup_window=output_cfg.get("dedup_window", 2.0))

        search_stats.candidates_found = len(valid_candidates)
        search_stats.runtime_seconds = round(time.time() - start_wall, 2)

        # Structured Diagnostic Trace Logging
        logger.info("============================================================")
        logger.info("CANDIDATE DIAGNOSTIC TRACE SUMMARY")
        logger.info("============================================================")
        for idx, cand in enumerate(top_candidates, 1):
            logger.info(f"QUERY:                '{target_text}'")
            if cand.asr_span:
                logger.info(f"ASR SEGMENT:          {cand.asr_span.segment_start:.2f}s -> {cand.asr_span.segment_end:.2f}s (text='{cand.asr_span.segment_text}')")
                logger.info(f"ASR ESTIMATED QUERY:  {cand.asr_span.query_start:.2f}s -> {cand.asr_span.query_end:.2f}s")
            else:
                logger.info(f"ASR ESTIMATED QUERY:  none")
            if target_search_window:
                logger.info(f"VISUAL SEARCH WINDOW: {target_search_window[0]:.2f}s -> {target_search_window[1]:.2f}s")
            if cand.visual_span:
                logger.info(f"VISUAL TRACK:         {cand.visual_span.start:.2f}s -> {cand.visual_span.end:.2f}s (track_id={cand.visual_span.track_id})")
                logger.info(f"OCR TEXT:             '{cand.visual_span.ocr_text}'")
                logger.info(f"OCR CONFIDENCE:       {cand.visual_span.ocr_confidence:.4f}")
                logger.info(f"QUERY SIMILARITY:     {cand.visual_span.query_similarity:.4f}")
            else:
                logger.info(f"VISUAL TRACK:         none")
            logger.info(f"DECISION:             speech_match={cand.speech_match}, visual_text_match={cand.visual_text_match}, sources={cand.sources}")
            logger.info(f"EVIDENCE TIMESTAMP:   {cand.timestamp:.3f}s (frame #{cand.frame_number})")
            logger.info("------------------------------------------------------------")

        # Format output contract
        formatted_results = []
        for cand in top_candidates:
            evidence_filename = f"evidence_{cand.timestamp:.3f}s.jpg"
            evidence_dest = output_dir / evidence_filename
            if cand.image_path and Path(cand.image_path).exists():
                shutil.copy2(cand.image_path, evidence_dest)
                img_rel_path = str(evidence_dest).replace("\\", "/")
            else:
                img_rel_path = ""

            match_level = determine_match_level(cand.fused_score or 0.0, sim_thresh, matching_cfg.get("high_match_threshold", 0.85))

            res_entry = {
                "timestamp": format_timestamp_hhmmss(cand.timestamp),
                "timestamp_seconds": cand.timestamp,
                "frame_number": cand.frame_number,
                "extracted_text": cand.text,
                "match_strength": cand.fused_score or 0.0,
                "match_level": match_level,
                "image_path": img_rel_path,
                "sources": cand.sources,
                "evidence": {
                    "speech_match": cand.speech_match,
                    "visual_text_match": cand.visual_text_match
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
