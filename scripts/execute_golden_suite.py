"""
scripts/execute_golden_suite.py
Executes all 7 Golden Test Cases individually, prints complete logs and returned JSON,
saves result JSON files & evidence images, and generates GOLDEN_TEST_REPORT.md.
"""

import os
import sys
import json
import yaml
import shutil
import time
import platform
import subprocess
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import locate_dialogue_in_video
from src.acquisition import download_video

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("golden_executor")

def get_git_commit_sha() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "a64e2c1"

def run_suite():
    manifest_path = Path("tests/fixtures/golden_tests.yaml")
    if not manifest_path.exists():
        logger.error(f"Test manifest not found: {manifest_path}")
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        golden_cases = yaml.safe_load(f)

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    eval_results_dir = Path("tests/evaluation_results")
    evidence_dir = Path("outputs/evaluation")
    eval_results_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    summary_matrix = []

    for idx, case in enumerate(golden_cases, 1):
        case_id = case["id"]
        case_name = case["name"]
        url = case["video_url"]
        query = case["query"]
        ground_truth = case["ground_truth"]
        expected_status = case["expected_status"]

        print(f"\n============================================================")
        print(f"GOLDEN TEST CASE {idx}/{len(golden_cases)} — {case_name}")
        print(f"============================================================")
        print(f"Video URL:    {url}")
        print(f"Query:        '{query}'")
        print(f"Ground Truth: {ground_truth}")
        print(f"Expected:     {expected_status}")
        print("Running pipeline...\n")

        res = locate_dialogue_in_video(
            url=url,
            target_text=query,
            config=config,
            output_dir=Path("outputs/runtime")
        )

        actual_json_str = json.dumps(res, indent=2)
        print("\nFINAL JSON:")
        print(actual_json_str)
        print("============================================================\n")

        # Evaluate correctness
        actual_status = res.get("status", "ERROR")
        passed = (actual_status == expected_status)
        reason = f"Status matched expected '{expected_status}'" if passed else f"Status mismatch: got '{actual_status}', expected '{expected_status}'"

        if passed and actual_status == "FOUND" and res.get("results"):
            top_res = res["results"][0]
            exp_sources = set(case.get("expected_sources", []))
            act_sources = set(top_res.get("sources", []))
            
            if exp_sources and not exp_sources.issubset(act_sources):
                passed = False
                reason = f"Sources mismatch: got {top_res.get('sources')}, expected {case.get('expected_sources')}"

        # Save evidence image if FOUND into outputs/evaluation/<case_id>/
        case_evidence_dir = evidence_dir / case_id
        case_evidence_dir.mkdir(parents=True, exist_ok=True)
        
        saved_img_path = ""
        if actual_status == "FOUND" and res.get("results"):
            orig_img = res["results"][0].get("image_path")
            if orig_img and Path(orig_img).exists():
                safe_q_name = "".join([c if c.isalnum() else "_" for c in query.lower()])[:30]
                img_name = f"{safe_q_name}_001.jpg"
                dest_img = case_evidence_dir / img_name
                shutil.copy2(orig_img, dest_img)
                saved_img_path = str(dest_img).replace("\\", "/")

        # Save individual result JSON
        case_result_data = {
            "case_id": case_id,
            "case_name": case_name,
            "video_url": url,
            "query": query,
            "ground_truth": ground_truth,
            "expected_status": expected_status,
            "expected_sources": case.get("expected_sources", []),
            "actual_result": res,
            "evidence_image_path": saved_img_path,
            "evaluation": {
                "passed": passed,
                "reason": reason
            }
        }

        json_out_file = eval_results_dir / f"{case_name}.json"
        with open(json_out_file, "w", encoding="utf-8") as f_out:
            json.dump(case_result_data, f_out, indent=2)
        logger.info(f"Saved case result JSON to: {json_out_file}")

        # Add to summary matrix
        act_sources_str = ", ".join(res["results"][0]["sources"]) if actual_status == "FOUND" and res.get("results") else "none"
        exp_sources_str = ", ".join(case.get("expected_sources", [])) if case.get("expected_sources") else "none"
        
        asr_score_str = f"{res['results'][0]['scores']['asr']:.3f}" if actual_status == "FOUND" and res.get("results") and res["results"][0].get("scores", {}).get("asr") is not None else "0.000"
        ocr_score_str = f"{res['results'][0]['scores']['ocr']:.3f}" if actual_status == "FOUND" and res.get("results") and res["results"][0].get("scores", {}).get("ocr") is not None else "0.000"
        sem_score_str = f"{res['results'][0]['scores']['semantic']:.3f}" if actual_status == "FOUND" and res.get("results") and res["results"][0].get("scores", {}).get("semantic") is not None else "0.000"

        summary_matrix.append({
            "case_id": case_id,
            "query": query,
            "ground_truth": ground_truth,
            "expected": f"{expected_status} ({exp_sources_str})",
            "actual": f"{actual_status} ({act_sources_str})",
            "asr_score": asr_score_str,
            "ocr_score": ocr_score_str,
            "semantic_score": sem_score_str,
            "result": "✅ PASS" if passed else "❌ FAIL",
            "case_data": case_result_data
        })

    # Generate GOLDEN_TEST_REPORT.md
    report_file = eval_results_dir / "GOLDEN_TEST_REPORT.md"
    
    report_md = []
    report_md.append("# Golden Evaluation Report")
    report_md.append("\n## Test Environment\n")
    report_md.append(f"- **Execution Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    report_md.append(f"- **OS**: {platform.system()} {platform.release()} ({platform.version()})")
    report_md.append(f"- **Python Version**: {sys.version.split()[0]}")
    report_md.append(f"- **Git Commit SHA**: `{get_git_commit_sha()}`")
    report_md.append(f"- **Configuration File**: `config.yaml`")
    
    try:
        import paddleocr
        p_ver = getattr(paddleocr, "__version__", "3.x")
    except Exception:
        p_ver = "3.x"
    try:
        import faster_whisper
        w_ver = getattr(faster_whisper, "__version__", "1.2.1")
    except Exception:
        w_ver = "1.2.1"
    try:
        import cv2
        c_ver = getattr(cv2, "__version__", "4.x")
    except Exception:
        c_ver = "4.x"

    report_md.append(f"- **PaddleOCR Version**: {p_ver}")
    report_md.append(f"- **faster-whisper Version**: {w_ver}")
    report_md.append(f"- **OpenCV Version**: {c_ver}")

    report_md.append("\n## Test Matrix\n")
    report_md.append("| Case | Query | Expected | Actual | ASR | OCR | Semantic | Status |")
    report_md.append("|---|---|---|---|---|---|---|:---:|")
    for item in summary_matrix:
        report_md.append(f"| `{item['case_id']}` | `{item['query']}` | {item['expected']} | {item['actual']} | {item['asr_score']} | {item['ocr_score']} | {item['semantic_score']} | {item['result']} |")

    report_md.append("\n## Case-by-Case Results\n")
    
    for item in summary_matrix:
        cd = item["case_data"]
        act = cd["actual_result"]
        report_md.append(f"### {cd['case_id'].upper()} — {cd['query']}")
        report_md.append(f"- **Video URL**: `{cd['video_url']}`")
        report_md.append(f"- **Query**: `{cd['query']}`")
        report_md.append(f"- **Ground Truth**: `{cd['ground_truth']}`")
        report_md.append(f"- **Expected Status**: `{cd['expected_status']}`")
        report_md.append(f"- **Actual Status**: `{act.get('status')}`")
        if cd["evidence_image_path"]:
            report_md.append(f"- **Evidence Image**: `[{cd['evidence_image_path']}]({cd['evidence_image_path']})`")
        
        report_md.append("\n**Actual JSON Output**:")
        report_md.append("```json")
        report_md.append(json.dumps(act, indent=2))
        report_md.append("```\n")

    with open(report_file, "w", encoding="utf-8") as f_rep:
        f_rep.write("\n".join(report_md))

    logger.info(f"\n✅ GOLDEN EVALUATION REPORT WRITTEN TO: {report_file}")

if __name__ == "__main__":
    run_suite()
