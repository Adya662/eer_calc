#!/usr/bin/env python3
import os, sys, time, shutil, subprocess
from pathlib import Path
from dotenv import load_dotenv
import datetime
import csv
from collections import Counter
import json

relevant_labels = {"PERSON", "ORG"}

env_path = Path(__file__).parent / ".env"
if env_path.exists(): load_dotenv(env_path)

stats = []
entity_type_counter = Counter()

def main(call_dir: str):
    call_start = time.time()
    start_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"🚀 Processing {call_dir} - start at {start_ts}")
    cd = Path(call_dir)
    out = cd/"output"; out.mkdir(exist_ok=True)
    script = Path(__file__).parent.parent/"eer"/"eer_spacy.py"
    ref_path = (cd / "ref_transcript.json").resolve()
    gt_path  = (cd / "gt_transcript.json").resolve()
    cmd = [sys.executable, str(script), str(ref_path), str(gt_path)]
    subprocess.run(cmd, cwd=out, check=True, env=os.environ)    

    # Load entity counts from result file
    eer_result_path = out / "eer_spacy_results.json"
    if not eer_result_path.exists():
        print(f"⚠️  Warning: Missing eer_spacy_results.json for {cd.name}")
        return
    with open(eer_result_path) as f:
        result = json.load(f)

    # Compute concerned entities separately for ref and gt based on relevant labels
    concerned_ref = []
    concerned_gt = []
    for pair in result.get("sample_error_pairs", []):
        for ent in pair.get("ref_entities", []):
            if ent.get("type") in relevant_labels:
                concerned_ref.append(ent["text"])
        for ent in pair.get("gt_entities", []):
            if ent.get("type") in relevant_labels:
                concerned_gt.append(ent["text"])
    # Dedupe
    concerned_ref = list(set(concerned_ref))
    concerned_gt = list(set(concerned_gt))

    ref_entities = []
    gt_entities = []
    for pair in result.get("sample_error_pairs", []):
        ref_entities.extend(ent["text"] for ent in pair.get("ref_entities", []))
        gt_entities.extend(ent["text"] for ent in pair.get("gt_entities", []))

    # Count all entity types across ref and gt
    for pair in result.get("sample_error_pairs", []):
        for ent in pair.get("ref_entities", []):
            entity_type_counter[ent.get("type")] += 1
        for ent in pair.get("gt_entities", []):
            entity_type_counter[ent.get("type")] += 1

    # Extract mispronounced IDs from Label Studio or annotated annotation
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
    # Dedupe
    pronunciation_ids = list(set(pronunciation_ids))
    # Log mispronunciation details
    print(f"🗣️  {cd.name} mispronunciation count: {len(pronunciation_ids)}, IDs: {pronunciation_ids}")

    stats.append({
        "call_id": cd.name,
        "ref_total_entities": len(ref_entities),
        "ref_unique_entities": len(set(ref_entities)),
        "gt_total_entities": len(gt_entities),
        "gt_unique_entities": len(set(gt_entities)),
        "concerned_ref_entities": len(concerned_ref),
        "concerned_gt_entities": len(concerned_gt),
        "mispronunciation_count": len(pronunciation_ids),
        "pronunciation_ids": ";".join(pronunciation_ids),
        "number_of_mispronounced_words": len(pronunciation_ids),
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
        fieldnames = [
            "call_id",
            "ref_total_entities",
            "ref_unique_entities",
            "gt_total_entities",
            "gt_unique_entities",
            "concerned_ref_entities",
            "concerned_gt_entities",
            "mispronunciation_count",
            "pronunciation_ids",
            "number_of_mispronounced_words",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats)

    print(f"📝 Entity summary written to {csv_path}")

    # Print summary of entity types
    print("📊 All entity types encountered across calls:")
    for ent_type, count in entity_type_counter.items():
        print(f" - {ent_type}: {count}")

    script_end = time.time()
    total_duration = script_end - script_start
    print(f"⏳ Total script runtime: {total_duration:.2f}s 🎉")