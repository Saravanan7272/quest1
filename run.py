"""
run.py
CLI Entry Point for Video Dialogue / Text Locator.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
import yaml

from src.pipeline import locate_dialogue_in_video

def main():
    parser = argparse.ArgumentParser(
        description="Video Dialogue / Text Locator using Query-Guided Temporal Search."
    )
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="Public video URL (e.g. YouTube, OK.ru, or equivalent)."
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target text string to search for in video dialogue."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed debug logging."
    )

    args = parser.parse_args()

    # Logging setup
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )

    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    else:
        logging.warning(f"Config file '{args.config}' not found. Using default empty config.")
        config = {}

    logging.info(f"Target Text: '{args.target}'")
    logging.info(f"Video URL:   '{args.url}'")

    result = locate_dialogue_in_video(
        url=args.url,
        target_text=args.target,
        config=config,
        output_dir=Path("outputs")
    )

    print("\n================ FINAL SEARCH RESULT ================")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("=====================================================\n")

    if result.get("status") == "ERROR":
        sys.exit(1)

if __name__ == "__main__":
    main()
