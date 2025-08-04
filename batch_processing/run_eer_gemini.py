#!/usr/bin/env python3
import os, sys, time, shutil, subprocess
from pathlib import Path
from dotenv import load_dotenv
import csv
import json

# Labels to consider for concerned entity counts
relevant_labels = {"PERSON", "ORG", "PRODUCT"}

env_path = Path(__file__).parent / ".env"
if env_path.exists(): load_dotenv(env_path)

total_start = time.time()
metrics_summary = []

def main(call_dir: str):
    start = time.time()
    cd = Path(call_dir)
    print(f"[{cd.name}] started at {time.strftime('%X')}")
    out = cd/"output"; out.mkdir(exist_ok=True)
    ref_path = cd / "ref_transcript.json"
    gt_path = cd / "gt_transcript.json"

    script = Path(__file__).parent.parent/"eer"/"eer_gemini.py"
    cmd = [
    sys.executable,
    str(script),
    str(ref_path.resolve()),    
    str(gt_path.resolve())
    ]
    subprocess.run(cmd, cwd=out, check=True, env=os.environ)

    src = out/"eer_gemini_results.json"
    if src.exists():
        with open(src, "r", encoding="utf-8") as f:
            results = json.load(f)
        ent = results.get("entity_summary", {})

        # Analyze concerned entities from sample error pairs
        sample_pairs = results.get("sample_error_pairs", [])
        concerned_ref_all = []
        concerned_gt_all = []
        for pair in sample_pairs:
            for e in pair.get("ref_entities", []):
                if e.get("type") in relevant_labels:
                    concerned_ref_all.append(e["text"])
            for e in pair.get("gt_entities", []):
                if e.get("type") in relevant_labels:
                    concerned_gt_all.append(e["text"])
        concerned_ref_entities = len(concerned_ref_all)
        concerned_gt_entities = len(concerned_gt_all)
        unique_concerned_ref_entities = len(set(concerned_ref_all))
        unique_concerned_gt_entities = len(set(concerned_gt_all))

        # Compute mispronunciation from annotations
        pronunciation_ids = []
        for file in cd.glob("*.json"):
            if file.name in ("label_studio.json", "annotated.json"):
                with open(file) as ls_file:
                    ls_data = json.load(ls_file)
                records = ls_data if isinstance(ls_data, list) else [ls_data]
                for record in records:
                    for ann in record.get("annotations", []):
                        for res in ann.get("result", []):
                            if res.get("from_name") == "pronunciation_note":
                                for txt in res.get("value", {}).get("text", []):
                                    id_part = txt.split("-", 1)[0].strip()
                                    if id_part:
                                        pronunciation_ids.append(id_part)
        pronunciation_ids = list(set(pronunciation_ids))
        mispronunciation_count = len(pronunciation_ids)
        pronunciation_ids_str = ";".join(pronunciation_ids)

        metrics_summary.append({
            "call_id": cd.name,
            "ref_total_entities": ent.get("total_ref_entities"),
            "ref_unique_entities": ent.get("unique_ref_entities"),
            "gt_total_entities": ent.get("total_gt_entities"),
            "gt_unique_entities": ent.get("unique_gt_entities"),
            "concerned_ref_entities": concerned_ref_entities,
            "concerned_gt_entities": concerned_gt_entities,
            "unique_concerned_ref_entities": unique_concerned_ref_entities,
            "unique_concerned_gt_entities": unique_concerned_gt_entities,
            "mispronunciation_count": mispronunciation_count,
            "pronunciation_ids": pronunciation_ids_str
        })

    elapsed = time.time() - start
    print(f"🕒 [{cd.name}] completed at {time.strftime('%X')}, took {elapsed:.2f}s")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: run_eer_gemini.py <call_dir or calls>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if input_path.is_dir():
        for subdir in sorted(input_path.glob("*")):
            if subdir.is_dir() and (subdir / "ref_transcript.json").exists() and (subdir / "gt_transcript.json").exists():
                try:
                    main(str(subdir))
                except subprocess.CalledProcessError as e:
                    print(f"[{subdir.name}] failed: {e}")
    else:
        print("Invalid directory. Provide a valid call folder or 'calls' directory.")
        sys.exit(1)

    total_time = time.time() - total_start
    print(f"\n✅ All calls processed in {total_time:.2f}s\n")

    with open("eer_gemini_metrics_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "call_id",
            "ref_total_entities", "ref_unique_entities",
            "gt_total_entities", "gt_unique_entities",
            "concerned_ref_entities", "concerned_gt_entities",
            "unique_concerned_ref_entities", "unique_concerned_gt_entities",
            "mispronunciation_count", "pronunciation_ids"
        ])
        writer.writeheader()
        writer.writerows(metrics_summary)
    print("📊 Wrote entity summary to eer_gemini_metrics_summary.csv")