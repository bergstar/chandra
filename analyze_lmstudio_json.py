import json
import re
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_ROOT / "output" / "lmstudio"
REPORT_PATH = OUTPUT_DIR / "analysis_report.json"
FILENAME_PATTERN = re.compile(r"^(\d{3})_(\d{5,6})_(\d{5,6})\.json$")
HEX_COLOR_PATTERN = re.compile(r"^#[0-9A-Fa-f]{6}$")
LATIN_PATTERN = re.compile(r"[A-Za-z]")
REQUIRED_ROOT_KEYS = {"participants", "screenshare", "speaking"}
REQUIRED_PARTICIPANT_KEYS = {
    "name",
    "speaking",
    "avatar_kind",
    "avatar_color",
    "avatar_initials",
}
ALLOWED_AVATAR_KINDS = {"initials", "photo", "illustration", "unknown"}
SEVERITY_SCORES = {
    "high": 100,
    "medium": 10,
    "low": 1,
}


def get_target_files(output_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in output_dir.glob("*.json")
        if path.is_file()
        and not path.name.startswith(".")
        and path.name != REPORT_PATH.name
    )


def parse_metadata(file_name: str) -> tuple[dict, list[dict]]:
    issues = []
    match = FILENAME_PATTERN.match(file_name)
    metadata = {
        "batch_id": None,
        "frame_id": None,
        "leaf_id": None,
    }
    if not match:
        issues.append(
            make_issue(
                "filename_format",
                "low",
                "Filename does not match the expected pattern <batch>_<frame>_<leaf>.json.",
            )
        )
        return metadata, issues

    metadata["batch_id"], metadata["frame_id"], metadata["leaf_id"] = match.groups()
    return metadata, issues


def make_issue(code: str, severity: str, message: str, **details) -> dict:
    issue = {
        "code": code,
        "severity": severity,
        "message": message,
    }
    if details:
        issue["details"] = details
    return issue


def add_issue(issues: list[dict], code: str, severity: str, message: str, **details):
    issues.append(make_issue(code, severity, message, **details))


def has_latin(text: str) -> bool:
    return bool(LATIN_PATTERN.search(text))


def finalize_entry(entry: dict):
    score = sum(SEVERITY_SCORES[issue["severity"]] for issue in entry["issues"])
    entry["severity_score"] = score

    severities = {issue["severity"] for issue in entry["issues"]}
    if "high" in severities:
        entry["status"] = "error"
    elif entry["issues"]:
        entry["status"] = "warning"
    else:
        entry["status"] = "ok"


def analyze_participants(participants, issues: list[dict]):
    participant_names = []
    speaking_true_names = []
    duplicate_names = []
    seen_names = Counter()

    if not isinstance(participants, list):
        add_issue(
            issues,
            "participants_not_list",
            "high",
            "Top-level 'participants' must be a list.",
            actual_type=type(participants).__name__,
        )
        return participant_names, speaking_true_names

    if not participants:
        add_issue(
            issues,
            "participants_empty",
            "medium",
            "Participants list is empty.",
        )
        return participant_names, speaking_true_names

    for index, participant in enumerate(participants):
        if not isinstance(participant, dict):
            add_issue(
                issues,
                "participant_not_object",
                "high",
                "Participant entry must be an object.",
                index=index,
                actual_type=type(participant).__name__,
            )
            continue

        missing_keys = sorted(REQUIRED_PARTICIPANT_KEYS - set(participant.keys()))
        if missing_keys:
            add_issue(
                issues,
                "participant_missing_keys",
                "high",
                "Participant object is missing required keys.",
                index=index,
                missing_keys=missing_keys,
            )

        name = participant.get("name")
        if not isinstance(name, str) or not name.strip():
            add_issue(
                issues,
                "participant_name_invalid",
                "high",
                "Participant name must be a non-empty string.",
                index=index,
                actual_value=name,
            )
            normalized_name = None
        else:
            normalized_name = name.strip()
            participant_names.append(normalized_name)
            seen_names[normalized_name] += 1
            if has_latin(normalized_name):
                add_issue(
                    issues,
                    "participant_name_latin",
                    "medium",
                    "Participant name contains Latin characters.",
                    index=index,
                    name=normalized_name,
                )

        participant_speaking = participant.get("speaking")
        if not isinstance(participant_speaking, bool):
            add_issue(
                issues,
                "participant_speaking_invalid",
                "high",
                "Participant 'speaking' must be a boolean.",
                index=index,
                actual_value=participant_speaking,
            )
        elif participant_speaking and normalized_name is not None:
            speaking_true_names.append(normalized_name)

        avatar_kind = participant.get("avatar_kind")
        if not isinstance(avatar_kind, str):
            add_issue(
                issues,
                "participant_avatar_kind_invalid_type",
                "high",
                "Participant 'avatar_kind' must be a string.",
                index=index,
                actual_value=avatar_kind,
            )
        elif avatar_kind not in ALLOWED_AVATAR_KINDS:
            add_issue(
                issues,
                "participant_avatar_kind_invalid_value",
                "medium",
                "Participant 'avatar_kind' is not one of the allowed values.",
                index=index,
                avatar_kind=avatar_kind,
            )

        avatar_color = participant.get("avatar_color")
        if not isinstance(avatar_color, str):
            add_issue(
                issues,
                "participant_avatar_color_invalid_type",
                "high",
                "Participant 'avatar_color' must be a string.",
                index=index,
                actual_value=avatar_color,
            )
        elif not HEX_COLOR_PATTERN.match(avatar_color):
            add_issue(
                issues,
                "participant_avatar_color_invalid_format",
                "medium",
                "Participant 'avatar_color' must be a #RRGGBB hex color.",
                index=index,
                avatar_color=avatar_color,
            )

        avatar_initials = participant.get("avatar_initials")
        if avatar_initials is not None and not isinstance(avatar_initials, str):
            add_issue(
                issues,
                "participant_avatar_initials_invalid_type",
                "high",
                "Participant 'avatar_initials' must be a string or null.",
                index=index,
                actual_value=avatar_initials,
            )
        elif (
            avatar_kind == "initials"
            and (not isinstance(avatar_initials, str) or not avatar_initials.strip())
        ):
            add_issue(
                issues,
                "participant_avatar_initials_missing",
                "medium",
                "Participants with avatar_kind='initials' should have avatar_initials.",
                index=index,
                name=normalized_name,
            )
        elif (
            isinstance(avatar_kind, str)
            and avatar_kind != "initials"
            and isinstance(avatar_initials, str)
            and avatar_initials.strip()
        ):
            add_issue(
                issues,
                "participant_avatar_initials_unexpected",
                "medium",
                "Participants without avatar_kind='initials' should not have avatar_initials.",
                index=index,
                name=normalized_name,
                avatar_initials=avatar_initials,
            )

    duplicate_names = sorted(name for name, count in seen_names.items() if count > 1)
    if duplicate_names:
        add_issue(
            issues,
            "participant_duplicate_names",
            "medium",
            "Duplicate participant names detected.",
            names=duplicate_names,
        )

    return participant_names, speaking_true_names


def analyze_root_object(root_object: dict, issues: list[dict]):
    actual_keys = set(root_object.keys())
    missing_keys = sorted(REQUIRED_ROOT_KEYS - actual_keys)
    extra_keys = sorted(actual_keys - REQUIRED_ROOT_KEYS)

    if missing_keys:
        add_issue(
            issues,
            "root_missing_keys",
            "high",
            "Top-level object is missing required keys.",
            missing_keys=missing_keys,
        )

    if extra_keys:
        add_issue(
            issues,
            "root_extra_keys",
            "low",
            "Top-level object contains unexpected keys.",
            extra_keys=extra_keys,
        )

    speaking = root_object.get("speaking")
    if not isinstance(speaking, list):
        add_issue(
            issues,
            "speaking_not_list",
            "high",
            "Top-level 'speaking' must be a list.",
            actual_type=type(speaking).__name__,
        )
        speaking_names = []
    else:
        speaking_names = []
        invalid_indices = []
        latin_entries = []
        for index, entry in enumerate(speaking):
            if not isinstance(entry, str):
                invalid_indices.append(index)
                continue
            normalized = entry.strip()
            speaking_names.append(normalized)
            if has_latin(normalized):
                latin_entries.append(normalized)
        if invalid_indices:
            add_issue(
                issues,
                "speaking_non_string_items",
                "high",
                "Top-level 'speaking' must contain only strings.",
                indices=invalid_indices,
            )
        if latin_entries:
            add_issue(
                issues,
                "speaking_names_latin",
                "medium",
                "Top-level 'speaking' contains names with Latin characters.",
                names=sorted(set(latin_entries)),
            )

    screenshare = root_object.get("screenshare")
    if not isinstance(screenshare, bool):
        add_issue(
            issues,
            "screenshare_not_bool",
            "high",
            "Top-level 'screenshare' must be a boolean.",
            actual_value=screenshare,
        )

    participant_names, speaking_true_names = analyze_participants(
        root_object.get("participants"),
        issues,
    )

    if speaking_names and participant_names:
        missing_from_participants = sorted(
            name for name in set(speaking_names) if name not in set(participant_names)
        )
        if missing_from_participants:
            add_issue(
                issues,
                "speaking_names_missing_from_participants",
                "medium",
                "Names listed in top-level 'speaking' are absent from participants.",
                names=missing_from_participants,
            )

    if speaking_true_names and speaking_names:
        participant_true_not_in_speaking = sorted(
            name for name in set(speaking_true_names) if name not in set(speaking_names)
        )
        if participant_true_not_in_speaking:
            add_issue(
                issues,
                "participant_true_not_in_speaking",
                "medium",
                "Participants marked speaking=true are absent from top-level 'speaking'.",
                names=participant_true_not_in_speaking,
            )


def analyze_file(json_path: Path) -> dict:
    metadata, filename_issues = parse_metadata(json_path.name)
    entry = {
        "file": json_path.name,
        "status": "ok",
        "severity_score": 0,
        "issues": list(filename_issues),
        "metadata": metadata,
    }

    raw_json = json_path.read_text(encoding="utf-8")
    if not raw_json.strip():
        add_issue(
            entry["issues"],
            "empty_file",
            "high",
            "File is empty.",
        )
        finalize_entry(entry)
        return entry

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        add_issue(
            entry["issues"],
            "invalid_json",
            "high",
            "File could not be parsed as JSON.",
            error=str(exc),
            line=exc.lineno,
            column=exc.colno,
        )
        finalize_entry(entry)
        return entry

    if not isinstance(data, list):
        add_issue(
            entry["issues"],
            "root_not_list",
            "high",
            "Top-level JSON value must be a list.",
            actual_type=type(data).__name__,
        )
        finalize_entry(entry)
        return entry

    if len(data) == 0:
        add_issue(
            entry["issues"],
            "root_empty_list",
            "high",
            "Top-level JSON list is empty.",
        )
        finalize_entry(entry)
        return entry

    if len(data) > 1:
        add_issue(
            entry["issues"],
            "root_multiple_items",
            "low",
            "Top-level JSON list contains more than one item.",
            item_count=len(data),
        )

    root_object = data[0]
    if not isinstance(root_object, dict):
        add_issue(
            entry["issues"],
            "root_first_item_not_object",
            "high",
            "The first item in the top-level list must be an object.",
            actual_type=type(root_object).__name__,
        )
        finalize_entry(entry)
        return entry

    analyze_root_object(root_object, entry["issues"])
    finalize_entry(entry)
    return entry


def build_report(files: list[dict]) -> dict:
    summary = {
        "total_files": len(files),
        "ok": sum(1 for entry in files if entry["status"] == "ok"),
        "warning": sum(1 for entry in files if entry["status"] == "warning"),
        "error": sum(1 for entry in files if entry["status"] == "error"),
    }
    return {
        "summary": summary,
        "files": files,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = get_target_files(OUTPUT_DIR)

    print("Analyze LM Studio JSON")
    print(f"Input: {OUTPUT_DIR}")
    print(f"Found {len(files)} file(s)")

    report_entries = [analyze_file(path) for path in files]
    report_entries.sort(key=lambda entry: entry["file"])
    report = build_report(report_entries)

    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Report: {REPORT_PATH}")
    print(
        "Summary:"
        f" ok={report['summary']['ok']}"
        f" warning={report['summary']['warning']}"
        f" error={report['summary']['error']}"
    )


if __name__ == "__main__":
    main()
