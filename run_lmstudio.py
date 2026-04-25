import base64
import json
import mimetypes
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parent
PROMPT_FILE = REPO_ROOT / "prompt.txt"
INPUT_DIR = REPO_ROOT / "input"
OUTPUT_DIR = REPO_ROOT / "output" / "lmstudio"
LMSTUDIO_CHAT_COMPLETIONS_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL = "chandra-ocr-2@f16"
MAX_TOKENS = 12384
TIMEOUT_SECONDS = 300


def get_supported_files(input_dir: Path) -> list[Path]:
    supported_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    files = []
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in supported_extensions:
            files.append(path)
            continue

        if not path.is_dir():
            continue

        for child in sorted(path.iterdir()):
            if child.is_file() and child.suffix.lower() in supported_extensions:
                files.append(child)

    return files


def load_prompt() -> str:
    prompt = PROMPT_FILE.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"Prompt file is empty: {PROMPT_FILE}")
    return prompt


def build_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(image_path.name)
    if mime_type is None:
        raise ValueError(f"Could not determine MIME type for {image_path}")

    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{image_b64}"


def build_payload(prompt: str, image_path: Path) -> dict:
    return {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": build_data_url(image_path),
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
        "max_tokens": MAX_TOKENS,
    }


def post_json(url: str, payload: dict) -> dict:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_model_output_content(response_json: dict) -> str:
    choices = response_json.get("choices", [])
    if not choices:
        raise ValueError("LM Studio response has no choices")

    message = choices[0].get("message", {})
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("LM Studio response has no assistant content")

    return content.strip()


def repair_output_content(output_content: str) -> str:
    try:
        from json_repair import repair_json
    except ImportError as exc:
        raise ImportError(
            "json-repair is required for run_lmstudio.py. Install it with: pip install json-repair"
        ) from exc

    return repair_json(output_content, ensure_ascii=False)


def get_output_dir(image_path: Path) -> Path:
    relative_parent = image_path.parent.relative_to(INPUT_DIR)
    return OUTPUT_DIR / relative_parent / image_path.stem


def save_outputs(image_path: Path, output_content: str) -> None:
    file_output_dir = get_output_dir(image_path)
    file_output_dir.mkdir(parents=True, exist_ok=True)

    output_path = file_output_dir / f"{image_path.stem}.json"
    output_path.write_text(output_content, encoding="utf-8")

    print(f"  Saved: {output_path}")


def save_error(image_path: Path, error_text: str) -> None:
    file_output_dir = get_output_dir(image_path)
    file_output_dir.mkdir(parents=True, exist_ok=True)
    error_path = file_output_dir / f"{image_path.stem}.error.txt"
    error_path.write_text(error_text, encoding="utf-8")
    print(f"  Saved: {error_path}")


def main() -> None:
    prompt = load_prompt()
    files = get_supported_files(INPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("LM Studio test runner")
    print(f"Input: {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Model: {MODEL}")
    print(f"Endpoint: {LMSTUDIO_CHAT_COMPLETIONS_URL}")
    print(f"Found {len(files)} file(s)")

    if not files:
        return

    for index, image_path in enumerate(files, start=1):
        relative_path = image_path.relative_to(INPUT_DIR)
        print(f"\n[{index}/{len(files)}] Processing: {relative_path}")
        payload = build_payload(prompt, image_path)
        try:
            response_json = post_json(LMSTUDIO_CHAT_COMPLETIONS_URL, payload)
            output_content = extract_model_output_content(response_json)
            repaired_content = repair_output_content(output_content)
            save_outputs(image_path, repaired_content)
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            error_text = f"HTTP {exc.code}\n{body}"
            save_error(image_path, error_text)
            print(f"  Error: HTTP {exc.code}")
        except URLError as exc:
            error_text = f"Connection error\n{exc}"
            save_error(image_path, error_text)
            print(f"  Error: {exc}")
        except Exception as exc:
            error_text = f"Unexpected error\n{exc}"
            save_error(image_path, error_text)
            print(f"  Error: {exc}")


if __name__ == "__main__":
    main()
