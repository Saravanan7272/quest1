"""
scripts/run_golden_suite.py
Automated runner script executing the 7-query Golden Benchmark Matrix.
"""

import sys
import logging
from pathlib import Path
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import locate_dialogue_in_video

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("golden_suite")

GOLDEN_BENCHMARK = [
    {
        "id": "Q1",
        "video": "tests/data/multimodal_sample.mp4",
        "target": "My mind rebels at stagnation",
        "expected_status": "FOUND",
        "expected_sources": ["asr", "ocr"],
        "expected_speech": True,
        "expected_visual": True
    },
    {
        "id": "Q2",
        "video": "https://youtu.be/dPTKl5H5ftg",
        "target": "My mind rebels at stagnation",
        "expected_status": "FOUND",
        "expected_sources": ["asr"],
        "expected_speech": True,
        "expected_visual": False
    },
    {
        "id": "Q3",
        "video": "tests/data/multimodal_sample.mp4",
        "target": "Batman",
        "expected_status": "NOT_FOUND",
        "expected_sources": [],
        "expected_speech": False,
        "expected_visual": False
    },
    {
        "id": "Q4",
        "video": "tests/data/multimodal_sample.mp4",
        "target": "The blue elephant is dancing",
        "expected_status": "NOT_FOUND",
        "expected_sources": [],
        "expected_speech": False,
        "expected_visual": False
    }
]

def run_golden_suite():
    with open("config.dev.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    passed_count = 0
    total_count = len(GOLDEN_BENCHMARK)

    logger.info(f"=== RUNNING GOLDEN BENCHMARK MATRIX ({total_count} QUERIES) ===")

    for item in GOLDEN_BENCHMARK:
        logger.info(f"\n--- Running {item['id']}: '{item['target']}' on '{item['video']}' ---")
        res = locate_dialogue_in_video(
            url=item["video"],
            target_text=item["target"],
            config=config,
            output_dir=Path("outputs")
        )

        status_ok = (res["status"] == item["expected_status"])
        
        if item["expected_status"] == "FOUND":
            top_res = res["results"][0]
            sources_ok = set(top_res["sources"]) == set(item["expected_sources"])
            speech_ok = (top_res["evidence"]["speech_match"] == item["expected_speech"])
            visual_ok = (top_res["evidence"]["visual_text_match"] == item["expected_visual"])
            item_passed = status_ok and sources_ok and speech_ok and visual_ok
            logger.info(
                f"Result: status={res['status']} (expected={item['expected_status']}), "
                f"sources={top_res['sources']} (expected={item['expected_sources']}), "
                f"speech_match={top_res['evidence']['speech_match']} (expected={item['expected_speech']}), "
                f"visual_text_match={top_res['evidence']['visual_text_match']} (expected={item['expected_visual']})"
            )
        else:
            item_passed = status_ok
            logger.info(f"Result: status={res['status']} (expected={item['expected_status']})")

        if item_passed:
            passed_count += 1
            logger.info(f"✅ {item['id']} PASSED")
        else:
            logger.error(f"❌ {item['id']} FAILED")

    logger.info(f"\n=== GOLDEN BENCHMARK SUMMARY: {passed_count}/{total_count} PASSED ===")
    assert passed_count == total_count, "Golden Benchmark Matrix regression test failed!"

if __name__ == "__main__":
    run_golden_suite()
