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

class ColoredFormatter(logging.Formatter):
    """Clean, professional log formatter with optional ANSI level highlighting."""
    CYAN = "\033[36m"
    GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    RED = "\033[1;31m"
    RESET = "\033[0m"

    def format(self, record):
        log_fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        if hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
            lvl = record.levelname
            if record.levelno == logging.INFO:
                if "SUCCESS" in record.getMessage() or "FOUND!" in record.getMessage():
                    lvl_str = f"{self.GREEN}SUCCESS{self.RESET}"
                else:
                    lvl_str = f"{self.CYAN}INFO{self.RESET}"
            elif record.levelno == logging.WARNING:
                lvl_str = f"{self.YELLOW}WARNING{self.RESET}"
            elif record.levelno >= logging.ERROR:
                lvl_str = f"{self.RED}ERROR{self.RESET}"
            else:
                lvl_str = record.levelname
            rec_copy = logging.makeLogRecord(record.__dict__)
            rec_copy.levelname = lvl_str
            return logging.Formatter(log_fmt, datefmt="%H:%M:%S").format(rec_copy)
        return logging.Formatter(log_fmt, datefmt="%H:%M:%S").format(record)

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

    # Suppress third-party verbose loggers
    for lib in ["httpx", "httpcore", "urllib3", "onnxruntime", "PIL", "matplotlib", "ppocr", "yt_dlp"]:
        logging.getLogger(lib).setLevel(logging.WARNING)

    # Logging setup with custom handler
    log_level = logging.DEBUG if args.verbose else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColoredFormatter())
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers = [handler]

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

    try:
        result = locate_dialogue_in_video(
            url=args.url,
            target_text=args.target,
            config=config,
            output_dir=Path("outputs")
        )
    except KeyboardInterrupt:
        logging.info("\nExecution cancelled by user (Ctrl+C). Exiting cleanly.")
        sys.exit(130)

    print("\n================ FINAL SEARCH RESULT ================")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("=====================================================\n")

    if result.get("status") == "ERROR":
        sys.exit(1)

if __name__ == "__main__":
    main()
