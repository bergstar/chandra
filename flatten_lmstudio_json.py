from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_ROOT / "output" / "lmstudio"


def get_nested_json_files(output_dir: Path) -> list[Path]:
    files = []
    for path in sorted(output_dir.rglob("*.json")):
        if path.parent == output_dir:
            continue
        if any(part.startswith(".") for part in path.relative_to(output_dir).parts):
            continue
        files.append(path)
    return files


def build_flat_name(json_path: Path) -> str:
    relative_path = json_path.relative_to(OUTPUT_DIR)
    stem_parts = list(relative_path.parts[:-1]) + [json_path.stem]
    return "_".join(stem_parts) + ".json"


def beautify_json(raw_json: str) -> str:
    try:
        from json_repair import repair_json
    except ImportError as exc:
        raise ImportError(
            "json-repair is required for flatten_lmstudio_json.py. Install it with: pip install json-repair"
        ) from exc

    return repair_json(raw_json, ensure_ascii=False, indent=2)


def main() -> None:
    files = get_nested_json_files(OUTPUT_DIR)

    print("Flatten LM Studio JSON")
    print(f"Source: {OUTPUT_DIR}")
    print(f"Found {len(files)} nested JSON file(s)")

    for index, json_path in enumerate(files, start=1):
        flat_name = build_flat_name(json_path)
        destination = OUTPUT_DIR / flat_name
        print(f"\n[{index}/{len(files)}] Processing: {json_path.relative_to(OUTPUT_DIR)}")

        raw_json = json_path.read_text(encoding="utf-8")
        beautified_json = beautify_json(raw_json)
        destination.write_text(beautified_json, encoding="utf-8")

        print(f"  Saved: {destination}")


if __name__ == "__main__":
    main()
