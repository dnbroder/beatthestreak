"""
update_readme_results.py
Reads summary.json + daily_log.json from S3 and updates the
Live Season Results section of README.md.
Runs daily via GitHub Actions after predictions fire.
"""
import boto3
import json
import re
from datetime import datetime

BUCKET = "mlb-hit-predictor-production-bucket"
s3 = boto3.client("s3")


def fetch(key):
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        return json.loads(obj["Body"].read())
    except Exception:
        return None


summary = fetch("predictions/latest/summary.json")
daily = fetch("predictions/latest/daily_log.json") or []

if not summary:
    print("No summary.json found — season not started yet, skipping update")
    exit(0)

updated = datetime.now().strftime("%Y-%m-%d %H:%M ET")

# Build metrics table
metrics = f"""| Metric | Value | Updated |
|--------|-------|---------|
| **Top-1 Hit Rate** | {summary.get('top1_hit_rate_pct', '—')}% | {updated} |
| **Current Streak** | {summary.get('current_streak', '—')} | {updated} |
| **Longest Streak** | {summary.get('longest_streak', '—')} | {updated} |
| **Dates Predicted** | {summary.get('dates_predicted', '—')} | {updated} |
| **Season Start** | March 27, 2026 | — |"""

# Build daily log (most recent 30, newest first)
rows = ""
for day in sorted(daily, key=lambda x: x["date"], reverse=True)[:30]:
    hit = day.get("top1_hit")
    if hit is True:
        result = "✅ HIT"
    elif hit is False:
        result = "❌ MISS"
    else:
        result = "⏳ Pending"
    streak = day.get("streak", "—")
    rows += (
        f"| {day['date']} | {day.get('top_pick', '—')} | "
        f"{day.get('team', '—')} | {day.get('confidence', '—')}% | "
        f"{result} | {streak} |\n"
    )

if not rows:
    rows = "| *Populates after Opening Day (March 27, 2026)* | | | | | |\n"

daily_table = f"""| Date | Top Pick | Team | Confidence | Result | Streak |
|------|----------|------|------------|--------|--------|
{rows}"""

# Update README
with open("README.md", "r") as f:
    readme = f.read()

readme = re.sub(
    r"(\| Metric \| Value \| Updated \|\n\|[-|]+\|\n(?:\|.*\|\n)+)",
    metrics + "\n",
    readme,
)
readme = re.sub(
    r"(\| Date \| Top Pick \| Team \| Confidence \| Result \| Streak \|\n\|[-|]+\|\n(?:\|.*\|\n)*)",
    daily_table + "\n",
    readme,
)

with open("README.md", "w") as f:
    f.write(readme)

print(
    f"README updated: {summary.get('dates_predicted', 0)} dates, "
    f"{summary.get('top1_hit_rate_pct', 0)}% hit rate, "
    f"streak={summary.get('current_streak', 0)}"
)