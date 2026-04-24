import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
PROMPT_FILE = REPO_ROOT / "prompt.txt"
INPUT_DIR = REPO_ROOT / "input"
OUTPUT_DIR = REPO_ROOT / "output"
METHOD = "hf"
MODEL_CHECKPOINT = "datalab-to/chandra-ocr-2"
MAX_OUTPUT_TOKENS = "12384"

os.environ["MODEL_CHECKPOINT"] = MODEL_CHECKPOINT
os.environ["MAX_OUTPUT_TOKENS"] = MAX_OUTPUT_TOKENS

from chandra.scripts.cli import main


def build_argv() -> list[str]:
    return [
        sys.argv[0],
        str(INPUT_DIR),
        str(OUTPUT_DIR),
        "--method",
        METHOD,
        "--prompt-file",
        str(PROMPT_FILE),
    ]


if __name__ == "__main__":
    sys.argv = build_argv()
    main()
