# Configuration Guide — Video Dialogue Locator (v1.0.0)

This document provides a parameter-by-parameter reference for `config.yaml` (Production Baseline) and `config.dev.yaml` (Fast Iteration Development).

---

## ⚙️ Configuration Reference Table

| Configuration Key | `config.yaml` Value | `config.dev.yaml` Value | Consumed In | Purpose & Impact |
|---|---|---|---|---|
| `asr.model` | `"base"` | `"tiny"` | `src/asr.py` | Whisper model size. `"base"` gives higher speech precision; `"tiny"` increases transcription speed. |
| `asr.device` | `"cpu"` | `"cpu"` | `src/asr.py` | PyTorch compute backend (`"cpu"` or `"cuda"`). |
| `asr.compute_type` | `"int8"` | `"int8"` | `src/asr.py` | Model quantization precision (`"int8"`, `"float16"`, `"float32"`). |
| `asr.window_padding` | `5.0` | `5.0` | `src/asr.py` | Temporal padding (seconds) added to candidate search window bounds. |
| `asr.candidate_threshold` | `0.50` | `0.50` | `src/asr.py` | Minimum ASR similarity score required to classify an audio candidate as FOUND. |
| `asr.partial_weight` | `0.70` | `0.70` | `src/matching.py` | Weight assigned to partial phrase fuzzy ratio in ASR scoring. |
| `asr.coverage_weight` | `0.30` | `0.30` | `src/matching.py` | Weight assigned to token coverage ratio in ASR scoring. |
| `sampling.coarse_fps` | `1.0` | `1.0` | `src/visual_pipeline.py` | FPS for initial coarse scout sampling across video. |
| `sampling.dense_fps` | `3.0` | `2.0` | `src/visual_pipeline.py` | FPS for dense text detection sampling within text-bearing windows. |
| `visual_scout.enabled` | `true` | `true` | `src/pipeline.py` | Master toggle for visual discovery path. |
| `visual_scout.threshold` | `30` | `30` | `src/visual_scout.py` | Pixel intensity change threshold for visual cut detection. |
| `visual_scout.sample_fps` | `1.0` | `1.0` | `src/visual_scout.py` | Frame rate for change detection scout. |
| `visual_scout.periodic_detection_fps` | `0.5` | `0.2` | `src/visual_scout.py` | Safety interval (FPS) for periodic text triggers. |
| `visual_scout.trigger_merge_window` | `2.0` | `3.0` | `src/visual_scout.py` | Temporal window (seconds) for merging adjacent scout triggers into clips. |
| `text_detector.min_confidence` | `0.30` | `0.30` | `src/text_detector.py` | Bounding box detection confidence threshold for RapidOCR detector. |
| `tracking.iou_threshold` | `0.5` | `0.5` | `src/text_tracker.py` | Minimum IoU overlap required to link text boxes across consecutive frames into a track. |
| `tracking.max_gap_seconds` | `0.5` | `0.5` | `src/text_tracker.py` | Maximum temporal gap allowed before closing an active text track. |
| `matching.character_weight` | `0.60` | `0.60` | `src/matching.py` | Character-level similarity weight for OCR text matching. |
| `matching.token_weight` | `0.40` | `0.40` | `src/matching.py` | Token coverage weight for OCR text matching. |
| `matching.similarity_threshold` | `0.75` | `0.75` | `src/pipeline.py` | High match threshold for candidate score filtering. |
| `matching.ocr_min_threshold` | `0.45` | `0.45` | `src/candidate_association.py` | Minimum OCR similarity required for candidate association. |
| `scoring.weights.asr` | `0.30` | `0.30` | `src/scoring.py` | Weight assigned to ASR score in score fusion. |
| `scoring.weights.ocr` | `0.50` | `0.50` | `src/scoring.py` | Weight assigned to OCR score in score fusion. |
| `scoring.weights.semantic` | `0.20` | `0.20` | `src/scoring.py` | Weight assigned to semantic embeddings score. |
| `output.top_k` | `3` | `3` | `src/pipeline.py` | Maximum number of top candidates returned in JSON response. |
| `output.dedup_window` | `2.0` | `2.0` | `src/pipeline.py` | Temporal deduplication window (seconds) for top-K candidates. |
