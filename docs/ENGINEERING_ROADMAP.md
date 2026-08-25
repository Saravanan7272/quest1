# Video Dialogue Locator — Engineering Roadmap & Future Development Strategy

---

## 1. Purpose of This Document

This document outlines the engineering roadmap and planned technical evolution for the **Video Dialogue Locator** beyond the **v1.0.0 production-oriented correctness baseline**.

### Key Principles
- **v1.0.0 Scoping Baseline**: The v1.0.0 release is a deliberately scoped correctness baseline engineered for high precision on dual-path dialogue retrieval (spoken audio speech and burned-in visual text).
- **Requirements-Driven Evolution**: Future architectural changes are not generic features; they must be driven by measured workload requirements, empirical benchmark evaluation, and scale constraints.
- **Living Strategy**: Future initiatives represent architectural extension points and potential evolution paths, not binding commitments to build every item without data justification.

### Requirement-Driven Engineering Loop

```mermaid
flowchart TD
    REQ["1. REQUIREMENT APPEARS"] --> MEASURE["2. MEASURE CURRENT BOTTLENECK"]
    MEASURE --> IDENTIFY["3. IDENTIFY FAILURE / LIMITATION"]
    IDENTIFY --> SELECT["4. SELECT ARCHITECTURAL CHANGE ONLY IF JUSTIFIED"]
    SELECT --> BENCHMARK["5. BENCHMARK CHANGE"]
    BENCHMARK --> DECISION{"Improved Result?"}
    DECISION -->|YES| ADOPT["6a. ADOPT CHANGE"]
    DECISION -->|NO| REJECT["6b. REJECT / REVISIT"]
```

### Baseline vs. Future Scope
- **What Exists Today**: Single-node CLI pipeline performing dual-path retrieval (faster-whisper ASR + Progressive Visual Search RapidOCR/PaddleOCR), multi-metric candidate association, missing-modality score fusion, and exact frame evidence extraction.
- **What Is Currently Sufficient**: CPU execution for individual video queries up to several minutes long. All 7 Golden Benchmark Cases produced the expected outcome in the current verification run, and 36 unit tests pass cleanly.
- **What Triggers Architectural Evolution**: Workload SLAs (e.g. batch queueing), repeat queries over massive video corpora (requiring offline persistent indexing), non-English audio/subtitles (requiring multilingual models), and paraphrased intent matching (requiring active vector semantic embeddings).

---

## 2. Current v1.0.0 Baseline Architecture

The current implementation in the `quest1` codebase follows a strict dual-path pipeline:

```mermaid
flowchart TD
    USER_INPUT["User Query + Video URL"]
    
    subgraph INGESTION["Stage 1: Acquisition (src/acquisition.py)"]
        YTDLP["yt-dlp Media Ingestion"]
        FFPROBE["FFprobe Metadata & Audio Stream Detection"]
        YTDLP --> FFPROBE
    end

    subgraph AUDIO_PATH["Stage 2: Audio Evidence Path (src/asr.py)"]
        WHISPER["faster-whisper ASR (int8 CPU)"]
        WORD_SPAN["find_query_span(): Word-Level Timestamps"]
        WHISPER --> WORD_SPAN
    end

    subgraph VISUAL_PATH["Stage 3: Progressive Visual Search (src/visual_pipeline.py)"]
        SCOUT["Visual Change Scout (1 FPS) + Periodic Triggers (0.5 FPS)"]
        TEXT_FILTER["Coarse Text Pre-Filter (TextDetector)"]
        DENSE_SAMPLING["Targeted Dense Sampling (3 FPS)"]
        TRACKING["Multi-Box IoU Tracker (TextTracker)"]
        OCR_EVAL["Representative Track OCR (OCREngineAdapter)"]
        
        SCOUT --> TEXT_FILTER --> DENSE_SAMPLING --> TRACKING --> OCR_EVAL
    end

    subgraph FUSION["Stage 4 & 5: Association, Fusion & Evidence Selection"]
        ASSOC["Candidate Association (src/candidate_association.py)"]
        SCORE_FUSE["Missing-Modality Score Fusion (src/scoring.py)"]
        DEDUP["Deduplication & Top-K Selection (src/pipeline.py)"]
        EVIDENCE["JSON Output Contract + Evidence Frame JPG"]
        
        ASSOC --> SCORE_FUSE --> DEDUP --> EVIDENCE
    end

    USER_INPUT --> INGESTION
    FFPROBE -->|Audio Stream| AUDIO_PATH
    FFPROBE -->|Visual Stream| VISUAL_PATH
    AUDIO_PATH -->|ASR Query Span| VISUAL_PATH
    AUDIO_PATH --> ASSOC
    VISUAL_PATH --> ASSOC
```

### Verified Modality Execution Modes
1. **Spoken-Only Dialogue (`sources = ["asr"]`)**: Cases 1, 2, 4. ASR isolates query phrase in audio; visual search finds no matching text. Returns ASR query start timestamp.
2. **Silent Visual Text (`sources = ["ocr"]`)**: Case 3. ASR fails or scores $< 0.50$; pipeline falls back to global visual search funnel, discovering on-screen captions via selective OCR.
3. **Multimodal Match (`sources = ["asr", "ocr"]`)**: Case 5. ASR isolates spoken audio span (`14.68s -> 15.68s`); visual search targets `[13.18s, 19.18s]`, locating text card at `15.347s`. Candidates are associated within 5.0s window and score-fused ($S_{\text{fused}} = 0.7499$).
4. **Negative Control (`status = "NOT_FOUND"`)**: Cases 6 & 7. Query is absent from both speech and text cards. Pipeline safely returns empty candidate results.

### Active Modality Scoring in v1.0.0

- **`scores.asr`**: Computed using a weighted combination of partial phrase similarity and token coverage:

  $$0.70 \cdot \text{partial\_ratio} + 0.30 \cdot \text{token\_coverage}$$

  Implemented in `src/matching.py` (`compute_asr_score`).

- **`scores.ocr`**: Computed using weighted lexical similarity and token coverage:

  $$0.60 \cdot \text{ratio} + 0.40 \cdot \text{token\_coverage}$$

  Implemented in `src/matching.py` (`compute_ocr_score`).

- **`scores.semantic`**: **Interface Extension Point**. The `ModalityScores` dataclass and JSON output schema include `semantic: Optional[float]`, but it returns `null` in v1.0.0 because semantic embeddings are disabled (`semantic.enabled = false` in `config.yaml`).

---

## 3. CURRENT → FUTURE Architecture Evolution

```mermaid
flowchart TD
    ROOT["VIDEO DIALOGUE LOCATOR"]
    
    subgraph BASELINE["v1.0.0 BASELINE (What exists today)"]
        direction TB
        B_CORRECT["Correctness Baseline"]
        B_EVAL["Evaluation (7 Cases, 36 Tests)"]
        B_CORE["Core Dual-Path Architecture"]
        B_RETR["ASR + OCR Retrieval & Score Fusion"]
        
        B_CORRECT --- B_RETR
        B_EVAL --- B_RETR
        B_CORE --- B_RETR
    end

    subgraph ROADMAP["EVOLUTION ROADMAP (What requirements trigger the next architecture)"]
        direction TB
        R_QUAL["Quality"]
        R_SCALE["Scale"]
        R_PROD["Product"]
        
        R_QUAL_DETAILS["Semantic Vector Retrieval<br/>Multilingual ASR & OCR<br/>Score Calibration<br/>Temporal Accuracy (t-IoU)"]
        R_SCALE_DETAILS["Persistent Video Indexing<br/>GPU Worker Acceleration<br/>Feature Caching<br/>Distributed Worker Queue"]
        R_PROD_DETAILS["REST Search API<br/>Interactive Web UI<br/>User Feedback Loop<br/>Multi-Video Corpus Search"]
        
        R_QUAL --- R_QUAL_DETAILS
        R_SCALE --- R_SCALE_DETAILS
        R_PROD --- R_PROD_DETAILS
    end

    ROOT --> BASELINE
    ROOT --> ROADMAP
```

---

## 4. Priority Level Definitions

- **P0 — Production Reliability & Core Correctness**: Crucial fixes or observability capabilities required before running unmonitored production workloads.
- **P1 — High-Value Engineering Improvements**: Improvements providing major precision/recall gains or substantial performance acceleration.
- **P2 — Product Expansion**: User-facing features, web interfaces, and API access layers expanding accessibility.
- **P3 — Large-Scale / Enterprise Evolution**: Distributed clustering, multi-tenant queuing, and offline index storage required only for massive throughput.

---

## 5. Requirements-Driven Development Matrix

| ID | Area | Initiative | Priority | Trigger / Requirement | Expected Benefit | Complexity |
| :--- | :--- | :--- | :---: | :--- | :--- | :---: |
| **RQ-01** | Quality | Semantic Vector Embedding Retrieval | **P1** | Poor recall on natural-language paraphrases | Matches conceptual meaning when exact phrasing differs | Medium |
| **RQ-02** | Quality | Multilingual ASR & OCR Expansion | **P2** | Non-English audio or multi-language subtitles | Extends retrieval to international media corpora | High |
| **EV-01** | Evaluation | Broad Evaluation Dataset & Harness | **P0** | Need statistical confidence beyond 7 cases | Automated precision/recall/F1 & temporal error tracking | Medium |
| **SC-01** | Scale | Persistent Video Feature Indexing | **P1** | Same video corpus receiving repeated queries | Target: sub-second candidate retrieval | High |
| **SC-02** | Performance| GPU Acceleration Pipeline | **P1** | Measured CPU throughput fails SLA requirements | Quantify inference speedup via CPU-vs-GPU benchmarks | Medium |
| **SC-03** | Scale | Distributed Worker Queue Architecture | **P3** | Ingestion workload exceeds single-node capacity | Horizontal scaling across multiple worker nodes | High |
| **OP-01** | Operations | Structured Observability & Telemetry | **P0** | Production deployment requirement | Distributed request tracing, metrics, and log aggregation | Low |
| **PR-01** | Product | REST Search API Layer | **P2** | Integration into third-party web apps | Standardized JSON HTTP API endpoint | Medium |
| **PR-02** | Product | Interactive Search Web Dashboard | **P2** | User requirement for visual timeline scrubbing | Visual inspection of evidence frames & tracks | Medium |

---

## 6. Detailed Technical Exploration Areas

### 6.1 Semantic Retrieval
- **Current Limitation**: Lexical matching (`compute_asr_score` & `compute_ocr_score`) relies on fuzzy edit distance and token overlap. If a query asks for *"Show me where they say goodbye"* but the dialogue speaks *"Farewell my friend"*, exact lexical matching fails.
- **Proposed Architecture**:
  ```text
  User Query ──> Sentence Transformer (all-MiniLM-L6-v2) ──> Query Embedding
                                                                   │
                                                   Cosine Similarity Calculation
                                                                   │
  ASR Transcript / OCR Box Text ──> Sentence Transformer ──> Text Embedding
  ```
- **Architectural Integration**: Activate `semantic.enabled: true` in `config.yaml`. Calculate cosine similarity between query and candidate text. Populate `scores.semantic` in `ModalityScores` and combine via `fuse_scores()`.

### 6.2 Multilingual Retrieval
- **Current Baseline**: `config.yaml` sets `ocr.lang: "en"` and `asr.language: null` (auto-detect English).
- **Future Requirements**: UTF-8 normalization, multi-language Whisper transcription, and PaddleOCR multi-language models (`lang: "ch"`, `lang: "es"`, `lang: "fr"`).

### 6.3 Temporal Localization Improvements
- **Current Baseline**: Computes ASR word span (`query_start`, `query_end`) and visual track best frame (`best_frame_timestamp`).
- **Future Improvements**: Evaluate temporal Intersection over Union ($\text{t-IoU}$) against annotated ground truth temporal intervals:
  $$\text{t-IoU} = \frac{\text{Overlap}(I_{\text{pred}}, I_{\text{gt}})}{\text{Union}(I_{\text{pred}}, I_{\text{gt}})}$$

### 6.4 OCR Robustness
- **Observed Failures**: Motion blur, stylized title fonts, low-contrast text cards, and rotated text.
- **Future Enhancements**: Image preprocessing pipeline (contrast enhancement via CLAHE, binarization) prior to passing crops to `OCREngineAdapter`.

### 6.5 Evaluation Expansion
- **Future Evaluation Metrics**:
  - **Recall@K**: Proportion of true dialogue occurrences located within top-K candidates.
  - **Mean Absolute Timestamp Error (MATE)**: $\frac{1}{N} \sum |t_{\text{pred}} - t_{\text{gt}}|$.
  - **Temporal IoU (t-IoU)**: Overlap ratio of predicted visual/speech interval against ground truth span.

---

## 7. Performance and Scaling Roadmap

```text
CURRENT WORKFLOW (v1.0.0 Execution):
Video URL ──> Download ──> ASR ──> Visual Scout ──> Dense Sampling ──> OCR ──> Result (25s - 269s)

FUTURE WORKFLOW (Offline Indexing + Fast Retrieval):
[Offline Ingestion] ──> Video ──> ASR Transcript Index ──┐
                              └──> Visual OCR Track Index ──┼──> Persistent Index (SQLite / Qdrant)
                                                            │
[Online Search]    ──> Query ───────────────────────────────┴──> Fast Index Query (Target: < 50ms)
```

- **Trigger for Persistent Indexing**: When repeated queries over the same video corpus make repeated ASR/OCR extraction more expensive than maintaining an offline feature index.

---

## 8. Configuration Parameter Reference & Taxonomy

Parameters in the system are categorized into three operational classes:

### A. Accuracy & Recall Parameters
- `asr.candidate_threshold` (Default: `0.50`): Minimum ASR candidate score required to establish audio match.
- `matching.ocr_min_threshold` (Default: `0.45`): Minimum OCR similarity score required for candidate association.
- `tracking.iou_threshold` (Default: `0.5`): Min Intersection over Union to link bounding boxes into persistent tracks.
- `tracking.max_gap_seconds` (Default: `0.5`): Max temporal gap before finalizing an inactive track.
- `sampling.dense_fps` (Default: `3.0`): Dense frame extraction rate; higher FPS improves short-text recall (< 0.5s).
- `visual_scout.periodic_detection_fps` (Default: `0.5`): Safety trigger frequency to capture static scene text.

### B. Search-Space & Compute Parameters
- `sampling.coarse_fps` (Default: `1.0`): Coarse sampling frequency for initial visual scout.
- `visual_scout.threshold` (Default: `30`): Pixel difference threshold for visual change detection.
- `visual_scout.trigger_merge_window` (Default: `2.0`): Temporal distance window for merging adjacent triggers.
- `output.top_k` (Default: `3`): Maximum candidate results returned in output payload.
- `output.dedup_window` (Default: `2.0`): Temporal window for deduplicating top candidate occurrences.

### C. Model & Environment Parameters
- `asr.model` (Default: `"base"`): Whisper ASR model selection (`"tiny"`, `"base"`, `"small"`, `"medium"`).
- `ocr.lang` (Default: `"en"`): OCR language recognition model target.
- `semantic.model` (Default: `"all-MiniLM-L6-v2"`): Sentence transformer model target for semantic embeddings.

---

## 9. Architecture Evolution by Workload

| Workload Scenario | Recommended Architecture | Primary Benefit |
| :--- | :--- | :--- |
| **Development & Testing** | `v1.0.0` CPU Pipeline (`config.dev.yaml`) | Fast local test execution without external dependencies |
| **Ad-Hoc Single Video Search** | `v1.0.0` Pipeline + Temp Caching | Zero setup overhead, clean single-pass execution |
| **Repeated Corpus Queries** | Offline Feature Extraction + Persistent Index | Target: sub-second or millisecond-scale candidate lookup |
| **High Throughput Batch Queue** | Celery / Redis Task Queue + GPU Workers | Horizontal scale across multiple processing nodes |

---

## 10. Product Evolution Lifecycle

```text
v1.0.0 (Current Baseline)    ──> CLI Correctness Baseline & 7 Golden Cases
v1.x (Reliability & Testing) ──> Structured Observability & Expanded Metrics
v2.0 (Product Access)        ──> REST API Service & Web Search Dashboard
v3.0 (Enterprise Search)     ──> Multi-Video Corpus Search & Semantic Vector Indexing
```

---

## 11. Enterprise & Operational Readiness Checklist

- [x] **Local Clean Cleanup**: Temp workspace cleaned via `shutil.rmtree()` in `finally:` block.
- [ ] **Structured Request Tracing**: Attach `request_id` (UUID) to all log statements across modules.
- [ ] **Resource Capping**: Enforce max memory and execution timeouts (`max_duration: 7200s` in `config.yaml`).
- [ ] **Containerization & Deployment**: Dockerfile packaging with locked system dependencies (`ffmpeg`, ONNX runtime).

---

## 12. Prioritization & Decision Matrix

| Initiative | User Value | Accuracy Impact | Performance Impact | Complexity | Risk | Trigger / Condition | Priority |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **Automated Metric Suite** | High | High | Low | Low | Low | Production benchmark expansion | **P0** |
| **Structured Telemetry & Logging** | High | Low | Low | Low | Low | Production deployment | **P0** |
| **GPU Inference Acceleration** | High | Medium | Very High | Medium | Low | CPU throughput SLA breaches | **P1** |
| **Persistent Video Indexing** | High | Low | Very High | High | Medium | Multi-query video corpus workload | **P1** |
| **Semantic Vector Retrieval** | High | High | Medium | Medium | Medium | Paraphrased query recall failures | **P1** |
| **REST API Service Endpoint** | Medium | Low | Low | Medium | Low | External application integration | **P2** |
| **Distributed Worker Queue** | Medium | Low | High | High | High | Single-node capacity saturation | **P3** |

---

## 13. Engineering Restraint: "What We Would NOT Build Yet"

To maintain architectural focus, the following capabilities are **explicitly deferred**:
1. **Microservices Architecture**: The system currently runs cleanly as a cohesive modular monolith. Splitting ASR and OCR into microservices would introduce network serialization latency without operational benefits for single-node processing.
2. **Real-time Streaming Media Analysis**: The pipeline is designed for stored video files. Real-time RTMP stream processing requires slide-window streaming inference, which is out of scope.
3. **General Object / Scene Recognition**: Object detection (e.g. YOLO) is excluded because the core problem statement is explicitly dialogue and text retrieval.

---

## 14. Recommended Phased Implementation Path

```text
Phase 1: Reliability & Observability (P0)
  ├── Implement Request IDs & OpenTelemetry logging
  └── Expand benchmark dataset to 50+ annotated video clips

Phase 2: Performance & Acceleration (P1)
  ├── Enable ONNX CUDA GPU acceleration for RapidOCR & Whisper
  └── Implement persistent SQLite/Qdrant video feature cache

Phase 3: Retrieval Enhancements (P1/P2)
  ├── Integrate Sentence-Transformers for semantic score fusion
  └── Implement multilingual OCR and ASR models

Phase 4: API & Product Dashboard (P2)
  ├── Build FastAPI REST endpoint
  └── Build Next.js / React interactive search UI
```

---

## 15. Architectural Decisions & Panel Defense

#### Q: "Why wasn't semantic vector search enabled in v1.0.0?"
> **Defense**: Exact lexical matching (fuzzy phrase ratio + token coverage) provides unambiguous ground-truth verification on dialogue text. Enabling semantic search prematurely risks introducing false positives from semantically related but unspoken phrases. We preserved `scores.semantic` in the dataclasses and scoring engine as an extension point, deferring activation until empirical benchmarks demonstrate a clear lexical recall gap.

#### Q: "Why do you use progressive visual search reduction instead of OCRing every frame?"
> **Defense**: A naïve frame-by-frame OCR architecture could require OCR evaluation on approximately 3,639 frames for a 121.31-second video. Our v1.0.0 5-stage reduction (Scout -> Pre-filter -> Dense Sampling -> Tracking -> Representative OCR) reduces OCR calls to 99 for Case 3 and 16 for Case 5—a 97.3% reduction in OCR calls—while verifying expected evidence timestamps (111.500s for Case 3, 15.347s for Case 5).

#### Q: "When would persistent indexing become necessary?"
> **Defense**: Persistent indexing is justified when repeated queries over the same video corpus make repeated ASR/OCR extraction more expensive than maintaining an offline feature index. Initial performance target: sub-second or millisecond-scale candidate retrieval from a persistent index, to be validated through benchmarking.

---

## 16. Requirement → Trigger → Architectural Response

| Requirement Appears | Evidence / Trigger | Architectural Response |
| :--- | :--- | :--- |
| **Paraphrased Queries** | Low recall on paraphrased evaluation dataset | Add semantic retrieval as an additional scoring signal (`scores.semantic`) |
| **Non-English Media** | Benchmark contains non-English audio or text | Integrate multilingual Whisper ASR and PaddleOCR language models |
| **Short-Lived Text Missed** | Recall failure on short visual text (< 0.5s duration) | Dynamically adapt temporal sampling density around candidate windows |
| **High CPU Latency** | Measured SLA violation during batch inference | Quantify GPU acceleration via controlled CPU-vs-GPU benchmarking |
| **Repeated Corpus Queries** | Repeated ASR/OCR re-computation over same videos | Introduce persistent feature index (SQLite / Qdrant) |
| **Concurrent Processing** | Single-node CPU/memory saturation | Transition to distributed worker queue architecture (Celery / Redis) |
| **External App Access** | Integration requirements beyond CLI interface | Expose REST Search API service endpoint |
| **Human Visual Review** | Users require visual timeline verification | Deploy interactive search UI and evidence viewer |
