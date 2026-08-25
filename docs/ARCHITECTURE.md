# System Architecture — Dual-Path Video Dialogue Locator (v1.0.0)

This document serves as the **authoritative deep technical architecture guide** for the **Video Dialogue Locator (v1.0.0 Baseline)**.

---

## 🏗️ High-Level Architectural Subsystems

```mermaid
flowchart TD
    subgraph INPUT["Input Query & Media"]
        U["User Query & Video URL"]
        ACQ["Video Acquisition\n(src/acquisition.py)"]
        U --> ACQ
    end

    subgraph DUAL_PATH["Dual-Path Retrieval Architecture"]
        direction TB
        subgraph AUDIO_PATH["Audio Evidence Path (ASR)"]
            ASR_TRANS["faster-whisper ASR\n(src/asr.py)"]
            WORD_SPAN["Word-Level Query Span\nfind_query_span()"]
            ASR_TRANS --> WORD_SPAN
            ASR_CAND["ASR Candidate\n(asr_query_start, end)"]
            WORD_SPAN --> ASR_CAND
        end

        subgraph VISUAL_PATH["Visual Discovery Path (src/visual_pipeline.py)"]
            SCOUT["Visual Scout & Triggers\n(src/visual_scout.py)"]
            PRE_FILTER["Coarse Text Pre-Filter\n(src/text_detector.py)"]
            DENSE_SAMPLING["Dense Sampling & IoU Tracking\n(src/text_tracker.py)"]
            OCR_EVAL["Representative Track OCR\n(src/ocr.py)"]
            SCOUT --> PRE_FILTER --> DENSE_SAMPLING --> OCR_EVAL
            VIS_CAND["Visual Candidate\n(visual_span, ocr_text)"]
            OCR_EVAL --> VIS_CAND
        end

        ACQ --> AUDIO_PATH
        ACQ --> VISUAL_PATH
    end

    subgraph FUSION["Candidate Association & Score Fusion"]
        ASSOC["Candidate Association\n(src/candidate_association.py)\nassociate_and_fuse_candidates()"]
        FUSE["Missing-Modality Score Fusion\n(src/scoring.py)"]
        ASR_CAND --> ASSOC
        VIS_CAND --> ASSOC
        ASSOC --> FUSE
    end

    subgraph OUTPUT["Evidence Selection & Output"]
        EVIDENCE["Modality-Aware Evidence Selection\n(src/pipeline.py)"]
        JSON_RES["JSON Contract &\nEvidence Image (jpg)"]
        FUSE --> EVIDENCE --> JSON_RES
    end
```

---

## 1. Stage 1 — Video Acquisition (`src/acquisition.py`)
- Downloads media stream via `yt-dlp` into `input_video.mp4` (`merge_output_format: mp4`, `retries: 10`).
- Inspects OpenCV video metadata: duration (seconds), FPS (frames/sec), resolution ($W \times H$), and `has_audio` (boolean).

---

## 2. Stage 2 — Audio Evidence Path (`src/asr.py`)
- Transcribes audio using `faster-whisper` (`word_timestamps=True`, `beam_size=5`).
- Computes ASR segment score combining fuzzy phrase ratio (70%) and token coverage (30%) via `compute_asr_score()` in `src/matching.py`.
- If segment score $\ge 0.50$, executes `find_query_span(words, target_text)` to extract precise word-level timestamps (`asr_query_start`, `asr_query_end`).
- Establishes targeted visual search window:
  $$\text{TargetedWindow} = [\max(0, \text{asr\_query\_start} - 1.5\text{s}), \min(\text{duration}, \text{asr\_query\_end} + 3.5\text{s})]$$

### Case 5 ASR Word-Level Timestamp Example:
Query: `"At least tell me your name"`
```text
Word        Start    End
------------------------
"At"        14.68s -> 14.86s
"least"     14.86s -> 15.00s
"tell"      15.00s -> 15.24s
"me"        15.24s -> 15.38s
"your"      15.38s -> 15.50s
"name"      15.50s -> 15.68s
------------------------
ASR Query Span: 14.68s -> 15.68s
```

---

## 3. Stage 3 — Progressive Visual Search Reduction (`src/visual_pipeline.py`)

The visual path uses **Progressive Visual Search Reduction** to avoid running CPU-expensive text recognition across thousands of raw video frames:

```mermaid
flowchart TD
    FULL["121.31s Total Video\n(~3,630 raw frames @ 30 FPS)"]
    WINDOW_SCOUT["1 FPS Scout + 0.5 FPS Periodic Triggers\n(61 merged trigger windows)"]
    TEXT_FILTER["Coarse Text Pre-Filter\n(6 text-bearing windows identified)"]
    DENSE_SAMPLES["3.0 FPS Dense Sampling\n(36 actual dense frames)"]
    TRACKING["Multi-Box IoU Tracking (threshold=0.5)\n(11 tracked text events)"]
    OCR_EVAL["Selective Track Frame & Crop Evaluation\n(99 OCR calls across representative frames)"]
    EXACT_MATCH["OCR Match: 'THANK YOU FOR WATCHING'\n(query_similarity = 1.0000)"]
    FINAL_EVIDENCE["Visual Evidence Frame: 111.500s\n(frame #3345)"]

    FULL -->|Coarse temporal scan| WINDOW_SCOUT
    WINDOW_SCOUT -->|Coarse text pre-filter| TEXT_FILTER
    TEXT_FILTER -->|Targeted dense sampling| DENSE_SAMPLES
    DENSE_SAMPLES -->|Bounding box IoU linking| TRACKING
    TRACKING -->|Candidate frame evaluation| OCR_EVAL
    OCR_EVAL -->|Fuzzy string matching| EXACT_MATCH
    EXACT_MATCH -->|Export JPG evidence| FINAL_EVIDENCE
```

### Stage Breakdown:
1. **Scout & Periodic Triggers (`src/visual_scout.py`)**: Computes frame difference triggers (1 FPS) and periodic safety triggers (0.5 FPS), merging adjacent triggers within `2.0s`.
2. **Coarse Text Pre-Filter (`src/text_detector.py`)**: Evaluates coarse scout frames with `TextDetector` to eliminate non-text regions (`Coarse text pre-filter: 6 text-bearing windows identified`).
3. **Dense Sampling (`src/sampling.py`)**: Samples remaining text-active windows at `sampling.dense_fps` (3.0 FPS in `config.yaml`).
4. **IoU Tracking (`src/text_tracker.py`)**: Links detected bounding boxes across consecutive frames into `TextEvent` tracks using Intersection over Union (`iou_threshold: 0.5`, `max_gap_seconds: 0.5`).
5. **Representative Track OCR (`src/ocr.py`)**: Samples candidate frames across track duration, runs OCR recognition, and dynamically sets `best_frame_timestamp` to the frame with the highest query similarity.

---

## 4. Stage 4 — Multi-Metric Candidate Association & Fusion (`src/candidate_association.py`)

Merges ASR query spans with visual track candidate spans using temporal compatibility AND multi-factor OCR evaluation:

```mermaid
flowchart TD
    ASR_CAND["ASR Candidate\n(14.68s -> 15.68s)"]
    VIS_CAND["Visual Track Candidate\n(13.18s -> 19.18s, OCR='ATLEASTTELL MEYOUR NAME')"]
    
    subgraph COMPAT_CHECK["Compatibility Verification"]
        TEMP_COMPAT["Temporal Overlap Check\n(v_start <= q_end + 5.0s AND v_end >= q_start - 5.0s)"]
        OCR_MIN["OCR Acceptance Policy\n(ocr_similarity >= 0.45 AND ocr_confidence >= 0.30)"]
    end

    subgraph SCORE_FUSION["Missing-Modality Score Fusion (src/scoring.py)"]
        FUSE["fuse_scores()\n(ASR weight=0.30, OCR weight=0.50)"]
    end

    ASR_CAND --> TEMP_COMPAT
    VIS_CAND --> TEMP_COMPAT
    TEMP_COMPAT --> OCR_MIN
    OCR_MIN --> FUSE
    FUSE --> MULTI_CAND["MULTIMODAL Candidate\n(sources=['asr', 'ocr'], fused_score=0.7499)"]
```

---

## 5. Stage 5 — Modality-Aware Evidence Selection (`src/pipeline.py`)

The pipeline chooses the final evidence timestamp and image path based on candidate sources:

- **Multimodal (`sources = ["asr", "ocr"]`)**: Selects the visual OCR frame timestamp (`visual_span.best_frame_timestamp` = `15.347s`).
- **OCR-Only (`sources = ["ocr"]`)**: Selects the visual OCR frame timestamp (`visual_span.best_frame_timestamp` = `111.500s`).
- **ASR-Only (`sources = ["asr"]`)**: Selects the ASR query start timestamp (`asr_span.query_start` = `87.660s`).

---

## 🚶 Comprehensive Modality Pipeline Walkthroughs

### 1. Spoken-Only Dialogue Pipeline (Case 1 / 2 / 4)

```mermaid
flowchart TD
    QUERY["Query: 'Why Do We Fall?'"]
    
    subgraph ASR_BRANCH["Audio Evidence Path (ASR)"]
        WHISPER["faster-whisper ASR (base model)"]
        WORD_SPAN["find_query_span(): Extract exact word timestamps\n(query_start = 87.660s, query_end = 89.200s)"]
        ASR_SCORE["ASR Match Score = 0.8173\n(70% Phrase Ratio + 30% Token Coverage)"]
        ASR_CAND["ASR Candidate (speech_match = true)"]
        WHISPER --> WORD_SPAN --> ASR_SCORE --> ASR_CAND
    end

    subgraph TARGETED_VISUAL["Targeted Visual Discovery Path"]
        WIN_BOUNDS["Targeted Search Window: [86.160s, 92.700s]"]
        VIS_DISCOVERY["Dense Sampling (3.0 FPS) & Text Detection"]
        NO_VIS_MATCH["No Visual Text Matched Query in Window\n(visual_text_match = false)"]
        WIN_BOUNDS --> VIS_DISCOVERY --> NO_VIS_MATCH
    end

    subgraph ASSOCIATION["Candidate Association & Fusion"]
        ASSOC["associate_and_fuse_candidates()\nTemporal Overlap Check"]
        SPOKEN_CAND["ASR-Only Candidate\n(sources = ['asr'], fused_score = 0.8173)"]
        ASR_CAND --> ASSOC
        NO_VIS_MATCH --> ASSOC
        ASSOC --> SPOKEN_CAND
    end

    subgraph EVIDENCE_SELECTION["Modality-Aware Evidence Selection"]
        ASR_EVIDENCE["Select ASR Query Span Timestamp\n(timestamp = 87.660s, frame #2630)"]
        EXPORT_JPG["Export Evidence Image: evidence_87.660s.jpg"]
        SPOKEN_CAND --> ASR_EVIDENCE --> EXPORT_JPG
    end

    QUERY --> ASR_BRANCH
    ASR_CAND -->|Guides search bounds| TARGETED_VISUAL
```

- **Query**: `"Why Do We Fall?"` (Case 1)
- **Audio Path**: `faster-whisper` transcribes audio and `find_query_span()` isolates the query word timestamps: `query_start = 87.660s`, `query_end = 89.200s` (ASR score = `0.8173`).
- **Visual Path**: Targeted visual window `[86.160s, 92.700s]` is sampled at 3.0 FPS. No visual text matching the query is found (`visual_text_match = false`).
- **Association**: `associate_and_fuse_candidates()` classifies candidate as Spoken-Only (`sources = ["asr"]`).
- **Evidence Selection**: Modality evidence logic selects `asr_span.query_start` timestamp (`87.660s`, frame #2630) and exports `outputs/runtime/evidence_87.660s.jpg`.

---

### 2. Silent Visual Text Pipeline (Case 3)

```mermaid
flowchart TD
    QUERY["Query: 'Thank you for watching'"]

    subgraph ASR_BRANCH["Audio Evidence Path (ASR)"]
        WHISPER["faster-whisper ASR"]
        ASR_REJECT["ASR Segment Score = 0.3932\n(< 0.50 candidate threshold)\nASR Candidate REJECTED"]
        WHISPER --> ASR_REJECT
    end

    subgraph GLOBAL_VISUAL["Global Visual Discovery Path"]
        FALLBACK["ASR Absent/Weak -> Fallback to Global Visual Scout"]
        SCOUT["Visual Scout (1 FPS) + Periodic Triggers (0.5 FPS)\n(61 merged trigger windows)"]
        PRE_FILTER["Coarse Text Pre-Filter\n(6 text-bearing windows identified)"]
        DENSE_SAMPLING["Dense Temporal Sampling @ 3.0 FPS\n(36 actual dense frames)"]
        TRACKING["Multi-Box IoU Tracking (threshold=0.5)\n(11 tracked text events)"]
        OCR_EVAL["Selective OCR Evaluation\n(99 OCR calls across representative frames)"]
        MATCH_FOUND["OCR Match: 'THANK YOU FOR WATCHING'\n(query_similarity = 1.0000, ocr_conf = 0.8868)"]

        FALLBACK --> SCOUT --> PRE_FILTER --> DENSE_SAMPLING --> TRACKING --> OCR_EVAL --> MATCH_FOUND
    end

    subgraph ASSOCIATION["Candidate Association & Fusion"]
        ASSOC["associate_and_fuse_candidates()"]
        VIS_CAND["Visual-Only Candidate\n(sources = ['ocr'], speech_match = false, visual_text_match = true)"]
        ASR_REJECT --> ASSOC
        MATCH_FOUND --> ASSOC
        ASSOC --> VIS_CAND
    end

    subgraph EVIDENCE_SELECTION["Modality-Aware Evidence Selection"]
        SELECT_VIS["Select Visual Track Best Frame Timestamp\n(timestamp = 111.500s, frame #3345)"]
        EXPORT_JPG["Export Evidence Image: evidence_111.500s.jpg"]
        VIS_CAND --> SELECT_VIS --> EXPORT_JPG
    end

    QUERY --> ASR_BRANCH
    QUERY --> GLOBAL_VISUAL
```

- **Query**: `"Thank you for watching"` (Case 3)
- **Audio Path**: Segment score = `0.3932` ($< 0.50$), ASR candidate is rejected.
- **Visual Path**: ASR candidate is absent $\rightarrow$ pipeline falls back to global visual scout (`0.0s` to `121.31s`).
  - **Coarse Scout**: 61 merged trigger windows.
  - **Coarse Text Pre-Filter**: Identifies 6 text-bearing windows.
  - **Dense Sampling**: 36 actual dense frames @ 3.0 FPS.
  - **IoU Tracking**: 11 tracked text events.
  - **Selective OCR**: 99 OCR calls across candidate track frames. Track 11 yields `"THANK YOU FOR WATCHING"` (query similarity = `1.0000`, OCR confidence = `0.8868`).
- **Association**: `associate_and_fuse_candidates()` classifies candidate as Visual-Only (`sources = ["ocr"]`).
- **Evidence Selection**: Modality evidence logic selects `visual_span.best_frame_timestamp` (`111.500s`, frame #3345) and exports `outputs/runtime/evidence_111.500s.jpg`.

---

### 3. Multimodal Pipeline — Spoken + Visual Text (Case 5)

```mermaid
flowchart TD
    QUERY["Query: 'At least tell me your name'"]
    
    subgraph ASR_BRANCH["ASR Branch"]
        WHISPER["faster-whisper ASR"]
        WORD_SPAN["find_query_span(): 14.68s -> 15.68s"]
        ASR_CAND["ASR Candidate (Score = 0.9500)"]
        WHISPER --> WORD_SPAN --> ASR_CAND
    end

    subgraph TARGETED_VISUAL["Targeted Visual Discovery"]
        SEARCH_WIN["Targeted Search Window\n[13.18s, 19.18s]"]
        SCOUT_DENSE["Scout + Dense Sampling @ 3.0 FPS"]
        TRACK_RECS["Track 1 & Track 2 (best_frame = 15.347s)"]
        OCR_REC["RapidOCR: 'ATLEASTTELL MEYOUR NAME'\n(query_sim = 0.7760)"]
        SEARCH_WIN --> SCOUT_DENSE --> TRACK_RECS --> OCR_REC
    end

    QUERY --> ASR_BRANCH
    ASR_CAND -->|Guides search bounds| TARGETED_VISUAL

    subgraph MULTIMODAL_FUSION["Candidate Association & Fusion"]
        ASSOC["associate_and_fuse_candidates()\nTemporal Overlap <= 5.0s AND ocr_conf >= 0.45"]
        MULTIMODAL["MULTIMODAL Candidate\n(sources = ['asr', 'ocr'], fused_score = 0.7499)"]
        ASR_CAND --> ASSOC
        OCR_REC --> ASSOC
        ASSOC --> MULTIMODAL
    end

    subgraph EVIDENCE_SELECTION["Evidence Frame Selection"]
        VIS_EVIDENCE["Select Visual OCR Frame\n(timestamp = 15.347s, frame #460)"]
        MULTIMODAL --> VIS_EVIDENCE
    end
```

- **Query**: `"At least tell me your name"` (Case 5)
- **Audio Path**: ASR detects spoken segment at `14.68s -> 15.68s` (ASR score = `0.95`).
- **Visual Path**: ASR candidate guides visual discovery to targeted window `[13.18s, 19.18s]`. 19 dense frames are sampled @ 3.0 FPS. Text detector and IoU tracker form 3 tracks. Stage 5 representative OCR evaluates candidate frames across tracks and selects `15.347s` (`"ATLEASTTELL MEYOUR NAME"`, query similarity = `0.7760`, OCR confidence = `0.827`).
- **Association**: `associate_and_fuse_candidates()` links ASR query span and visual track span (temporal diff = `0.667s` $\le 5.0\text{s}$, OCR sim = `0.7760` $\ge 0.45$), yielding a Multimodal candidate (`sources = ["asr", "ocr"]`, `fused_score = 0.7499`).
- **Evidence Selection**: Modality evidence logic selects `visual_span.best_frame_timestamp` (`15.347s`, frame #460) and exports `outputs/runtime/evidence_15.347s.jpg`.

---

### 4. Negative Control Pipeline — `NOT_FOUND` (Case 6 / 7)

```mermaid
flowchart TD
    QUERY["Query: 'Batman'"]

    subgraph ASR_BRANCH["Audio Evidence Path (ASR)"]
        WHISPER["faster-whisper ASR"]
        ASR_LOW["ASR Segment Score = 0.3500\n(< 0.50 threshold)\nASR Candidate REJECTED"]
        WHISPER --> ASR_LOW
    end

    subgraph GLOBAL_VISUAL["Global Visual Discovery Path"]
        SCOUT["Visual Scout (1 FPS) + Periodic Triggers (0.5 FPS)\n(13 coarse trigger windows)"]
        PRE_FILTER["Coarse Text Pre-Filter\n(13 text-bearing windows)"]
        DENSE["Dense Sampling @ 3.0 FPS\n(75 dense frames)"]
        TRACK["IoU Text Tracking\n(10 tracked text events)"]
        OCR_EVAL["Selective OCR Evaluation\n(55 OCR calls)"]
        OCR_DISCOVERED["OCR Discovered Text Cards:\n- 'KINGOFMINDSET' (sim = 0.1263)\n- 'BUT IT'S NOT' (sim = 0.2118)\n- 'WAIT' (sim = 0.2400)\n- 'AT LEAST TELL ME YOUR NAME' (sim = 0.1655)"]
        
        SCOUT --> PRE_FILTER --> DENSE --> TRACK --> OCR_EVAL --> OCR_DISCOVERED
    end

    subgraph SIMILARITY_CHECK["Relevance Threshold Evaluation"]
        REJECT_ALL["All OCR Similarity Scores <= 0.2400\n(< 0.45 ocr_min_threshold)\nVisual Candidate REJECTED"]
        OCR_DISCOVERED --> REJECT_ALL
    end

    subgraph FINAL_OUTPUT["Pipeline Response Contract"]
        NOT_FOUND_RES["status = 'NOT_FOUND'\nresults = []\ntotal_candidates = 0"]
        ASR_LOW --> NOT_FOUND_RES
        REJECT_ALL --> NOT_FOUND_RES
    end

    QUERY --> ASR_BRANCH
    QUERY --> GLOBAL_VISUAL
```

- **Query**: `"Batman"` (Case 6)
- **Audio Path**: Whisper ASR segment score = `0.3500` ($< 0.50$), ASR candidate is rejected.
- **Visual Path**: Global visual scout scans video. OCR extracts discovered text cards (`"KINGOFMINDSET"`, `"BUT IT'S NOT"`, `"WAIT"`, `"AT LEAST TELL ME YOUR NAME"`). Fuzzy similarity matching yields max query similarity = `0.2400` ($< 0.45\text{ threshold}$). Visual candidate is rejected.
- **Association**: Zero valid candidates survive score threshold filtering.
- **Output**: `locate_dialogue_in_video()` returns `status = "NOT_FOUND"`, `results = []`, `total_candidates = 0`.

