#!/usr/bin/env python3
import os, json, re, subprocess, time
from pathlib import Path
from dotenv import load_dotenv

# ─── Setup ──────────────────────────────────────────────────────────────────
env_path = Path(__file__).parent / ".env"
if env_path.exists(): load_dotenv(env_path)

def generate_reference_transcript(json_path: Path, output_path: Path) -> None:
    import datetime
    def format_ts(seconds):
        return str(datetime.timedelta(seconds=float(seconds))) if seconds is not None else None
    def load_tasks(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    tasks = load_tasks(json_path)
    transcript = []
    for task in tasks:
        annotations = task.get("annotations", [])
        for ann in annotations:
            results = ann.get("result", [])
            speaker_map = {}
            for r in results:
                if r.get("type") == "labels" and r.get("from_name") == "speaker_labels":
                    start = r["value"].get("start")
                    if start is not None:
                        rounded = round(float(start), 2)
                        speaker = r["value"].get("labels", [None])[0]
                        if speaker:
                            speaker_map[rounded] = speaker
            for r in results:
                if r.get("type") == "textarea" and r.get("from_name") == "transcription":
                    start = r["value"].get("start")
                    texts = r["value"].get("text", [])
                    rounded = round(float(start), 2) if start is not None else None
                    speaker = speaker_map.get(rounded, "Unknown")
                    timestamp = format_ts(start)
                    for text in texts:
                        transcript.append({
                            "timestamp": timestamp,
                            "speaker": speaker,
                            "content": text.strip()
                        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

# ─── Functionality ─────────────────────────────────────────────────────────
def split_annotations(input_file: str, output_dir: str = "calls"):
    start_all = time.time()
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    base = Path(output_dir)
    base.mkdir(exist_ok=True)

    # Extract call_id from audio URL
    rx = re.compile(r"/calls/\d{4}/\d{2}/\d{2}/([0-9a-fA-F-]+)/")
    grouped = {}
    for item in data:
        url = item.get("data", {}).get("audio", "")
        m = rx.search(url)
        if m:
            cid = m.group(1)
            grouped.setdefault(cid, []).append(item)

    for cid, items in grouped.items():
        t0 = time.time()
        d = base/cid
        d.mkdir(exist_ok=True)
        # save annotations
        (d/"annotated.json").write_text(json.dumps(items, indent=2), encoding="utf-8")
        # download transcript.json
        audio_url = items[0]["data"]["audio"]
        transcript_url = re.sub(r"call_audio\.(?:mp4|ogg)$", "transcript.json", audio_url)
        out = d/"ref_transcript.json"
        subprocess.run(["aws","s3","cp", transcript_url, str(out)], check=True)
        # generate gt_transcript.json from annotated.json
        ann_path = d / "annotated.json"
        gt_path = d / "gt_transcript.json"
        generate_reference_transcript(ann_path, gt_path)
        elapsed = time.time()-t0
        print(f"[{cid}] split+download took {elapsed:.2f}s")

    total = time.time()-start_all
    print(f"All calls prepared in {total:.2f}s")


if __name__=="__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("input_file", help="combined annotations.json")
    p.add_argument("--output_dir", default="calls")
    args = p.parse_args()
    split_annotations(args.input_file, args.output_dir)