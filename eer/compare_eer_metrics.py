import json
import csv
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "output"
JSON_FILES = [
    "spacy_results.json",
    "stanford_results.json",
    "gemini_results.json",
    "presidio_results.json"
]
CSV_PATH = OUTPUT_DIR / "eer_metrics_comparison.csv"

FIELDS = [
    "method",
    "missed_entities",
    "total_entities",
    "percentage",
    "reference_entities",
    "gtothesis_entities",
    "unique_ref_entities",
    "unique_gt_entities"
]

rows = []
for fname in JSON_FILES:
    fpath = OUTPUT_DIR / fname
    if not fpath.exists():
        continue
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    method = data.get("method", fname.replace("_results.json", ""))
    metrics = data.get("metrics", {}).get("EER", {})
    summary = data.get("entity_summary", {})
    row = [
        method,
        metrics.get("missed_entities", ""),
        metrics.get("total_entities", ""),
        metrics.get("percentage", ""),
        summary.get("reference_entities", ""),
        summary.get("gtothesis_entities", ""),
        summary.get("unique_ref_entities", ""),
        summary.get("unique_gt_entities", "")
    ]
    rows.append(row)

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(FIELDS)
    writer.writerows(rows)

print(f"[✔] Wrote comparison CSV to {CSV_PATH}")