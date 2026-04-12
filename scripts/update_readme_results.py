"""
update_readme_results.py

Reads summary.json + daily_log.json from S3 and updates the Live Season
Results section of README.md in place.

Runs daily via GitHub Actions at 1 PM ET after the morning actuals checker
and prediction trigger have finished populating S3.

Design: uses HTML comment anchors in README.md to define exact replacement
regions. Original version used regex matching on table column headers,
which was fragile — any structural edit to the README between the anchors
and the tables could cause the regex to match the wrong block or consume
unrelated content. Comment anchors are invisible in rendered markdown but
explicit for the script.

Anchors required in README.md:
    <!-- BTS:RUNNING_RECORD_START -->
    ...table content managed by this script...
    <!-- BTS:RUNNING_RECORD_END -->

    <!-- BTS:DAILY_LOG_START -->
    ...table content managed by this script...
    <!-- BTS:DAILY_LOG_END -->

If either anchor pair is missing, the script logs an error and exits
without modifying the README. That's safer than guessing.
"""

import boto3
import json
import re
import sys
from datetime import datetime

BUCKET = "mlb-hit-predictor-production-bucket"
README_PATH = "README.md"
DAILY_LOG_DAYS_TO_SHOW = 30

# Anchor comment pairs — must match exactly in README.md
RUNNING_RECORD_START = "<!-- BTS:RUNNING_RECORD_START -->"
RUNNING_RECORD_END = "<!-- BTS:RUNNING_RECORD_END -->"
DAILY_LOG_START = "<!-- BTS:DAILY_LOG_START -->"
DAILY_LOG_END = "<!-- BTS:DAILY_LOG_END -->"

s3 = boto3.client("s3")


def fetch(key: str):
    """Fetch a JSON object from S3. Returns None if missing or unreadable."""
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        return json.loads(obj["Body"].read())
    except Exception as e:
        print(f"WARNING: fetch failed for {key}: {e}")
        return None


def fmt_pct(val) -> str:
    """
    Format a hit-rate percentage. Treats 0 as a real value (displays '0.0%')
    so early-season or cold-streak stats render honestly rather than as a
    placeholder. The only placeholder is the string '—' itself, which we pass
    through unchanged for days where the pipeline hasn't recorded anything.
    """
    if val is None or val == "—":
        return "—"
    try:
        return f"{float(val):.1f}%"
    except (TypeError, ValueError):
        return "—"


def hit_cell(val) -> str:
    """Render a hit/miss/pending cell."""
    if val is True:
        return "✅"
    if val is False:
        return "❌"
    return "⏳"


def build_running_record(summary: dict, updated: str) -> str:
    """Render the Running Record table as markdown, including anchor lines."""
    rows = [
        RUNNING_RECORD_START,
        "| Metric | Value | Updated |",
        "|--------|-------|---------|",
        f"| **Top-1 Hit Rate** | {fmt_pct(summary.get('top1_hit_rate_pct'))} | {updated} |",
        f"| **Top-2 Hit Rate** | {fmt_pct(summary.get('top2_hit_rate_pct'))} | {updated} |",
        f"| **Top-5 Hit Rate** | {fmt_pct(summary.get('top5_hit_rate_pct'))} | {updated} |",
        f"| **Top-10 Hit Rate** | {fmt_pct(summary.get('top10_hit_rate_pct'))} | {updated} |",
        f"| **Current Streak (Top-1)** | {summary.get('current_streak', '—')} | {updated} |",
        f"| **Longest Streak (Top-1)** | {summary.get('longest_streak', '—')} | {updated} |",
        f"| **Dates Predicted** | {summary.get('dates_predicted', '—')} | {updated} |",
        f"| **Season Start** | March 25, 2026 | — |",
        RUNNING_RECORD_END,
    ]
    return "\n".join(rows)


def build_daily_log(daily: list) -> str:
    """Render the Daily Log table as markdown, including anchor lines."""
    header = [
        DAILY_LOG_START,
        "| Date | Top Pick | Team | Confidence | Top-1 | Top-2 | Top-5 | Top-10 | Streak |",
        "|------|----------|------|------------|-------|-------|-------|--------|--------|",
    ]

    # Most recent N days, newest first
    recent = sorted(daily, key=lambda x: x["date"], reverse=True)[:DAILY_LOG_DAYS_TO_SHOW]

    if not recent:
        body = [
            "| *Populates after Opening Day (March 25, 2026)* | | | | | | | | |",
        ]
    else:
        body = []
        for day in recent:
            conf = day.get("confidence", "—")
            conf_str = f"{conf}%" if conf != "—" else "—"
            body.append(
                f"| {day['date']} "
                f"| {day.get('top_pick', '—')} "
                f"| {day.get('team', '—')} "
                f"| {conf_str} "
                f"| {hit_cell(day.get('top1_hit'))} "
                f"| {hit_cell(day.get('top2_hit'))} "
                f"| {hit_cell(day.get('top5_hit'))} "
                f"| {hit_cell(day.get('top10_hit'))} "
                f"| {day.get('streak', '—')} |"
            )

    return "\n".join(header + body + [DAILY_LOG_END])


def replace_region(readme: str, start_anchor: str, end_anchor: str, new_block: str) -> str:
    """
    Replace everything between start_anchor and end_anchor (inclusive) with
    new_block. The new_block should itself contain the anchors. If anchors
    aren't found, returns None — the caller decides how to handle it.
    """
    pattern = re.compile(
        re.escape(start_anchor) + r".*?" + re.escape(end_anchor),
        re.DOTALL,
    )
    if not pattern.search(readme):
        return None
    return pattern.sub(new_block, readme)


def main() -> int:
    summary = fetch("predictions/latest/summary.json")
    daily = fetch("predictions/latest/daily_log.json") or []

    if not summary:
        print("No summary.json found — season not started yet, skipping update")
        return 0

    updated = datetime.now().strftime("%Y-%m-%d %H:%M ET")

    try:
        with open(README_PATH, "r") as f:
            readme = f.read()
    except FileNotFoundError:
        print(f"ERROR: {README_PATH} not found")
        return 1

    # Verify both anchor pairs exist before touching anything
    missing = []
    for anchor in (RUNNING_RECORD_START, RUNNING_RECORD_END, DAILY_LOG_START, DAILY_LOG_END):
        if anchor not in readme:
            missing.append(anchor)
    if missing:
        print("ERROR: missing required anchor comments in README.md:")
        for a in missing:
            print(f"  {a}")
        print("Add these HTML comments around the managed tables before re-running.")
        return 2

    # Replace running record
    new_record = build_running_record(summary, updated)
    readme = replace_region(readme, RUNNING_RECORD_START, RUNNING_RECORD_END, new_record)
    if readme is None:
        print("ERROR: could not replace running record region")
        return 3

    # Replace daily log
    new_log = build_daily_log(daily)
    readme = replace_region(readme, DAILY_LOG_START, DAILY_LOG_END, new_log)
    if readme is None:
        print("ERROR: could not replace daily log region")
        return 4

    with open(README_PATH, "w") as f:
        f.write(readme)

    print(
        f"README updated: {summary.get('dates_predicted', 0)} dates, "
        f"top1={summary.get('top1_hit_rate_pct', 0)}%, "
        f"top2={summary.get('top2_hit_rate_pct', 0)}%, "
        f"top5={summary.get('top5_hit_rate_pct', 0)}%, "
        f"top10={summary.get('top10_hit_rate_pct', 0)}%, "
        f"streak={summary.get('current_streak', 0)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())