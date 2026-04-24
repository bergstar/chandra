import sys
from pathlib import Path

from chandra.scripts.cli import main


def inject_default_prompt_file(argv: list[str]) -> list[str]:
    if "--prompt-file" in argv or any(
        arg.startswith("--prompt-file=") for arg in argv
    ):
        return argv

    prompt_path = Path(__file__).resolve().parent / "prompt.txt"
    return [*argv, "--prompt-file", str(prompt_path)]


if __name__ == "__main__":
    sys.argv = [sys.argv[0], *inject_default_prompt_file(sys.argv[1:])]
    main()
