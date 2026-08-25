# Result JSON Schema Contract — Video Dialogue Locator (v1.0.0)

This document specifies the exact JSON response contract returned by `locate_dialogue_in_video()` in `src/pipeline.py`.

---

## 📋 JSON Payload Specification

```json
{
  "status": "FOUND | NOT_FOUND | ERROR",
  "results": [
    {
      "timestamp": "HH:MM:SS.sss",
      "timestamp_seconds": 15.347,
      "frame_number": 460,
      "extracted_text": "ATLEASTTELL MEYOUR NAME",
      "match_strength": 0.7499,
      "match_level": "HIGH | MEDIUM | LOW",
      "image_path": "outputs/evidence_15.347s.jpg",
      "sources": ["asr", "ocr"],
      "evidence": {
        "speech_match": true,
        "visual_text_match": true
      },
      "scores": {
        "asr": 0.95,
        "ocr": 0.6299,
        "semantic": null
      }
    }
  ],
  "total_candidates": 1,
  "search_summary": {
    "scout_frames": 10,
    "detector_frames": 4,
    "ocr_calls": 3,
    "candidates_found": 1,
    "tracked_events": 2,
    "runtime_seconds": 24.85
  }
}
```

---

## 💡 Payload Examples by Modality Mode

### Example 1: Multimodal Match (ASR + Visual OCR) — Case 5
```json
{
  "status": "FOUND",
  "results": [
    {
      "timestamp": "00:00:15.347",
      "timestamp_seconds": 15.347,
      "frame_number": 460,
      "extracted_text": "ATLEASTTELL MEYOUR NAME",
      "match_strength": 0.7499,
      "match_level": "HIGH",
      "image_path": "outputs/evidence_15.347s.jpg",
      "sources": ["asr", "ocr"],
      "evidence": {
        "speech_match": true,
        "visual_text_match": true
      },
      "scores": {
        "asr": 0.95,
        "ocr": 0.6299,
        "semantic": null
      }
    }
  ],
  "total_candidates": 1
}
```

### Example 2: Visual-Only Match (Silent Text) — Case 3
```json
{
  "status": "FOUND",
  "results": [
    {
      "timestamp": "00:01:51.500",
      "timestamp_seconds": 111.5,
      "frame_number": 3345,
      "extracted_text": "THANK YOU FOR WATCHING",
      "match_strength": 0.5,
      "match_level": "MEDIUM",
      "image_path": "outputs/evidence_111.500s.jpg",
      "sources": ["ocr"],
      "evidence": {
        "speech_match": false,
        "visual_text_match": true
      },
      "scores": {
        "asr": null,
        "ocr": 1.0,
        "semantic": null
      }
    }
  ],
  "total_candidates": 1
}
```

### Example 3: Spoken-Only Match — Case 1
```json
{
  "status": "FOUND",
  "results": [
    {
      "timestamp": "00:00:01.500",
      "timestamp_seconds": 1.5,
      "frame_number": 45,
      "extracted_text": "Why do we fall?",
      "match_strength": 0.3,
      "match_level": "LOW",
      "image_path": "outputs/evidence_1.500s.jpg",
      "sources": ["asr"],
      "evidence": {
        "speech_match": true,
        "visual_text_match": false
      },
      "scores": {
        "asr": 1.0,
        "ocr": null,
        "semantic": null
      }
    }
  ],
  "total_candidates": 1
}
```

### Example 4: Negative Control — Case 6
```json
{
  "status": "NOT_FOUND",
  "results": [],
  "total_candidates": 0
}
```
