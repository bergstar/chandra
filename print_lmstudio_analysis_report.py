import argparse
import json
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
REPORT_PATH = REPO_ROOT / "output" / "lmstudio" / "analysis_report.json"


def load_report(report_path: Path) -> dict:
    if not report_path.exists():
        raise FileNotFoundError(
            f"Analysis report not found: {report_path}\nRun `python analyze_lmstudio_json.py` first."
        )
    return json.loads(report_path.read_text(encoding="utf-8"))


def collect_issue_counts(entries: list[dict]) -> Counter:
    counter = Counter()
    for entry in entries:
        for issue in entry.get("issues", []):
            counter[issue["code"]] += 1
    return counter


def filter_entries(entries: list[dict], statuses: set[str]) -> list[dict]:
    return [entry for entry in entries if entry.get("status") in statuses]


def sort_entries(entries: list[dict]) -> list[dict]:
    return sorted(
        entries,
        key=lambda entry: (
            {"error": 0, "warning": 1, "ok": 2}.get(entry.get("status"), 3),
            -entry.get("severity_score", 0),
            entry.get("file", ""),
        ),
    )


def print_summary(report: dict):
    summary = report["summary"]
    print("LM Studio Analysis Report")
    print(f"Report: {REPORT_PATH}")
    print(
        "Summary:"
        f" total={summary['total_files']}"
        f" ok={summary['ok']}"
        f" warning={summary['warning']}"
        f" error={summary['error']}"
    )


def print_issue_summary(entries: list[dict]):
    issue_counts = collect_issue_counts(entries)
    if not issue_counts:
        print("\nIssue counts: none")
        return

    print("\nIssue counts:")
    for code, count in issue_counts.most_common(12):
        print(f"  {code}: {count}")


def summarize_issues(entry: dict, max_codes: int = 4) -> str:
    counts = Counter(issue["code"] for issue in entry.get("issues", []))
    top_codes = counts.most_common(max_codes)
    summary = ", ".join(f"{code}×{count}" for code, count in top_codes)
    remaining = len(counts) - len(top_codes)
    if remaining > 0:
        summary += f", +{remaining} more"
    return summary


def print_entries(
    entries: list[dict],
    limit: int | None = None,
    verbose: bool = False,
):
    if limit is not None:
        entries = entries[:limit]

    if not entries:
        print("\nFlagged files: none")
        return

    print("\nFlagged files:")
    for entry in entries:
        status_marker = {
            "error": "E",
            "warning": "W",
            "ok": "O",
        }.get(entry["status"], "?")
        print(
            f"{status_marker} {entry['file']} "
            f"score={entry['severity_score']} "
            f"issues={len(entry['issues'])} "
            f"[{summarize_issues(entry)}]"
        )
        if not verbose:
            continue

        metadata = entry.get("metadata", {})
        if any(metadata.values()):
            print(
                "  meta:"
                f" batch={metadata.get('batch_id')}"
                f" frame={metadata.get('frame_id')}"
                f" leaf={metadata.get('leaf_id')}"
            )
        for issue in entry.get("issues", []):
            print(f"  - [{issue['severity']}] {issue['code']}: {issue['message']}")
            details = issue.get("details")
            if details:
                detail_text = ", ".join(
                    f"{key}={value}" for key, value in sorted(details.items())
                )
                print(f"    {detail_text}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Print a human-readable console report for LM Studio JSON analysis."
    )
    parser.add_argument(
        "--status",
        choices=["all", "error", "warning", "ok", "flagged"],
        default="flagged",
        help="Which file statuses to print. Default: flagged (error + warning).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of printed file entries.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full per-issue details instead of the compact one-line view.",
    )
    args = parser.parse_args()

    report = load_report(REPORT_PATH)
    print_summary(report)

    statuses = {
        "all": {"ok", "warning", "error"},
        "error": {"error"},
        "warning": {"warning"},
        "ok": {"ok"},
        "flagged": {"warning", "error"},
    }[args.status]

    entries = sort_entries(filter_entries(report["files"], statuses))
    print_issue_summary(entries)
    print_entries(entries, limit=args.limit, verbose=args.verbose)


if __name__ == "__main__":
    main()
