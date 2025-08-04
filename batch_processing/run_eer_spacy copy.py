#!/usr/bin/env python3
import os, sys, time, shutil, subprocess
from pathlib import Path
from dotenv import load_dotenv
import datetime
import csv
from collections import Counter
import json

env_path = Path(__file__).parent / ".env"
if env_path.exists(): load_dotenv(env_path)

stats = []

def main(call_dir: str):
    call_start = time.time()
    start_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"🚀 Processing {call_dir} - start at {start_ts}")
    cd = Path(call_dir)
    out = cd/"output"; out.mkdir(exist_ok=True)
    for name in ("ref_transcript.json","gt_transcript.json"):
        shutil.copy(cd/name, out/name)

    script = Path(__file__).parent.parent/"eer"/"eer_spacy.py"
    cmd = [sys.executable, str(script), "ref_transcript.json", "gt_transcript.json"]
    subprocess.run(cmd, cwd=out, check=True, env=os.environ)

    # Load entity counts from result file
    with open(out / "eer_spacy_results.json") as f:
        result = json.load(f)

    ref_entities = result.get("ref_entities", [])
    gt_entities = result.get("gt_entities", [])

    stats.append({
        "call_id": cd.name,
        "ref_total_entities": len(ref_entities),
        "ref_unique_entities": len(set(ref_entities)),
        "gt_total_entities": len(gt_entities),
        "gt_unique_entities": len(set(gt_entities)),
    })

    call_end = time.time()
    end_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    duration = call_end - call_start
    print(f"✅ [{cd.name}] ended at {end_ts}")
    print(f"🕒 [{cd.name}] duration: {duration:.2f}s 😊")

if __name__=="__main__":
    if len(sys.argv)!=2:
        print("Usage: run_eer_spacy.py <calls_root_dir>")
        sys.exit(1)
    calls_root = sys.argv[1]
    script_start = time.time()
    for entry in Path(calls_root).iterdir():
        if entry.is_dir():
            main(str(entry))
    # Write CSV summary
    csv_path = Path(calls_root) / "entity_summary.csv"
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = ["call_id", "ref_total_entities", "ref_unique_entities", "gt_total_entities", "gt_unique_entities"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats)

    print(f"📝 Entity summary written to {csv_path}")
    script_end = time.time()
    total_duration = script_end - script_start
    print(f"⏳ Total script runtime: {total_duration:.2f}s 🎉")