import os
import json
from pathlib import Path
import re
import subprocess
import shutil
import sys
from dotenv import load_dotenv


# Load environment variables from .env file in batch_processing directory
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"[INFO] Loaded environment variables from {env_path}")
else:
    print(f"[WARNING] No .env file found at {env_path}")

# Verify API keys loaded
if os.getenv("OPENAI_API_KEY"):
    print("[INFO] OPENAI_API_KEY is set")
else:
    print("[WARNING] OPENAI_API_KEY is NOT set")
if os.getenv("GOOGLE_API_KEY"):
    print("[INFO] GOOGLE_API_KEY is set")
else:
    print("[WARNING] GOOGLE_API_KEY is NOT set")


from typing import Union
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def generate_reference_transcript(json_path: Union[str, Path], output_path: Union[str, Path] = None) -> None:
    import json
    import datetime
    from pathlib import Path

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

            # Build a speaker label map based on rounded start times
            speaker_map = {}
            for r in results:
                if r.get("type") == "labels" and r.get("from_name") == "speaker_labels":
                    start = r["value"].get("start")
                    if start is not None:
                        rounded = round(float(start), 2)
                        speaker = r["value"].get("labels", [None])[0]
                        if speaker:
                            speaker_map[rounded] = speaker

            # Collect transcription lines
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

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)


def split_annotations(input_file: str, output_dir: str = "calls"):
    """
    Reads a combined label_stdio.json file, groups annotations by call ID (extracted from the S3 audio URL),
    writes out one annotated JSON per call in its own folder, and downloads the corresponding transcript.json
    from S3 into each call's folder using AWS CLI.
    """
    # Load combined annotations
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Prepare output directory
    base_dir = Path(output_dir)
    base_dir.mkdir(exist_ok=True)

    # Regex to extract call ID
    call_id_pattern = re.compile(r"/calls/\d{4}/\d{2}/\d{2}/([0-9a-fA-F-]+)/")

    # Group by call ID
    grouped = {}
    audio_by_call = {}
    for item in data:
        audio_url = item.get("data", {}).get("audio", "")
        match = call_id_pattern.search(audio_url)
        if not match:
            continue
        call_id = match.group(1)
        grouped.setdefault(call_id, []).append(item)
        # store first audio_url for this call
        audio_by_call.setdefault(call_id, audio_url)

    # Report count
    num_calls = len(grouped)
    print(f"Found {num_calls} unique call(s) to extract")

    # Process each call
    for call_id, annotations in grouped.items():
        call_dir = base_dir / call_id
        call_dir.mkdir(exist_ok=True)
        # Write annotated.json
        ann_path = call_dir / "annotated.json"
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump(annotations, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(annotations)} annotations to {ann_path}")

        # Download ref_transcript.json using AWS CLI
        # 1) Load the annotated.json and extract the audio URL
        ann_items = json.loads(ann_path.read_text(encoding="utf-8"))
        audio_url = None
        for it in ann_items:
            audio_url = it.get("data", {}).get("audio")
            if audio_url:
                break
        if not audio_url:
            print(f"[ERROR] No audio URL found in {ann_path}")
            continue

        # 2) Construct the S3 path for transcript.json
        # Handle both .mp4 and .ogg audio filenames
        transcript_url = re.sub(r"call_audio\.(?:mp4|ogg)$", "transcript.json", audio_url)
        out_transcript = call_dir / "ref_transcript.json"

        # 3) Run AWS CLI to download
        cmd = [
            "aws", "s3", "cp",
            transcript_url,
            str(out_transcript)
        ]
        try:
            subprocess.run(cmd, check=True)
            print(f"[✔] Downloaded transcript to {out_transcript}")
            generate_reference_transcript(ann_path, call_dir / "gt_transcript.json")
        except subprocess.CalledProcessError:
            print(f"🔴 [ERROR] Failed to download transcript for call ID {call_id}")
            print(f"        🔴 Attempted S3 path: {transcript_url}")


def run_wer_eer_scripts(call_dir):
    """
    Run all WER and EER scripts for a given call directory.
    Assumes ref_transcript.json and gt_transcript.json are present in call_dir.
    Outputs are written inside call_dir/output/.
    """
    import subprocess
    from pathlib import Path
    import os

    call_dir = Path(call_dir)
    ref = call_dir / "ref_transcript.json"
    gt = call_dir / "gt_transcript.json"
    output_dir = call_dir / "output"
    output_dir.mkdir(exist_ok=True)

    # Copy transcripts to output dir for all scripts
    shutil.copy(ref, output_dir / "ref_transcript.json")
    shutil.copy(gt, output_dir / "gt_transcript.json")

    # Set up environment with API keys
    env = os.environ.copy()
    # Ensure subprocess gets the loaded API keys
    env['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY", "")
    env['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY", "")
    # Ensure we have the required API keys
    if not env.get('OPENAI_API_KEY'):
        print(f"[WARNING] OPENAI_API_KEY not found in environment for {call_dir.name}")
    if not env.get('GOOGLE_API_KEY'):
        print(f"[WARNING] GOOGLE_API_KEY not found in environment for {call_dir.name}")

    # WER scripts
    wer_scripts = [
        ("wer1.py", ["ref_transcript.json", "gt_transcript.json"]),
        ("wer2.py",),
        ("wer3.py",)
    ]
    
    # EER scripts
    eer_scripts = [
        ("eer_gemini.py", ["ref_transcript.json", "gt_transcript.json"]),
        ("eer_presidio.py", ["ref_transcript.json", "gt_transcript.json"]),
        ("eer_spacy.py", ["ref_transcript.json", "gt_transcript.json"]),
        ("eer_stanford.py", ["ref_transcript.json", "gt_transcript.json"])
    ]

    # Helper to run a script and move outputs to correct location
    def run_script_and_move_outputs(script, args=None, cwd=None):
        script_path = None
        if script.startswith("wer"):
            script_path = PROJECT_ROOT / "wer" / script
        else:
            script_path = PROJECT_ROOT / "eer" / script
        
        cmd = [sys.executable, str(script_path)]
        if args:
            cmd += args
        
        print(f"    Running: {' '.join(cmd)}")
        
        # Run the script with proper error handling
        try:
            result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True, env=env)
            if result.stdout:
                print(f"    Output: {result.stdout[:200]}...")
        except subprocess.CalledProcessError as e:
            print(f"    [ERROR] Script {script} failed with exit code {e.returncode}")
            print(f"    Error output: {e.stderr}")
            return False
        
        # Move outputs to the correct location based on script type
        moved = False
        if script == "wer1.py":
            # Move from global wer output to local
            global_output = Path("/Users/adyasrivastava/Metrics_Eval/wer/output/wer1_eval.json")
            local_output = output_dir / "wer1_eval.json"
            if global_output.exists():
                shutil.move(str(global_output), str(local_output))
                moved = True
                print(f"    Moved {script} output to {local_output}")
            else:
                print(f"    [WARNING] Expected output file {global_output} not found for {script}")
        elif script == "wer2.py":
            # Move from global wer output to local
            global_output = Path("/Users/adyasrivastava/Metrics_Eval/wer/output/wer2_eval.json")
            local_output = output_dir / "wer2_eval.json"
            if global_output.exists():
                shutil.move(str(global_output), str(local_output))
                moved = True
                print(f"    Moved {script} output to {local_output}")
            else:
                print(f"    [WARNING] Expected output file {global_output} not found for {script}")
        elif script == "wer3.py":
            # Move from global wer output to local
            global_output = Path("/Users/adyasrivastava/Metrics_Eval/wer/output/wer3_eval.json")
            local_output = output_dir / "wer3_eval.json"
            if global_output.exists():
                shutil.move(str(global_output), str(local_output))
                moved = True
                print(f"    Moved {script} output to {local_output}")
            else:
                print(f"    [WARNING] Expected output file {global_output} not found for {script}")
        elif script == "eer_gemini.py":
            # Move from current directory to local
            global_output = cwd / "eer_gemini_results.json"
            local_output = output_dir / "eer_gemini_results.json"
            if global_output.exists():
                shutil.move(str(global_output), str(local_output))
                moved = True
                print(f"    Moved {script} output to {local_output}")
            else:
                print(f"    [WARNING] Expected output file {global_output} not found for {script}")
        elif script == "eer_presidio.py":
            # Move from current directory to local
            global_output = cwd / "eer_presidio_results.json"
            local_output = output_dir / "eer_presidio_results.json"
            if global_output.exists():
                shutil.move(str(global_output), str(local_output))
                moved = True
                print(f"    Moved {script} output to {local_output}")
            else:
                print(f"    [WARNING] Expected output file {global_output} not found for {script}")
        elif script == "eer_spacy.py":
            # Move from current directory to local
            global_output = cwd / "eer_spacy_results.json"
            local_output = output_dir / "eer_spacy_results.json"
            if global_output.exists():
                shutil.move(str(global_output), str(local_output))
                moved = True
                print(f"    Moved {script} output to {local_output}")
            else:
                print(f"    [WARNING] Expected output file {global_output} not found for {script}")
        elif script == "eer_stanford.py":
            # Move from current directory to local
            global_output = cwd / "eer_stanford_results.json"
            local_output = output_dir / "eer_stanford_results.json"
            if global_output.exists():
                shutil.move(str(global_output), str(local_output))
                moved = True
                print(f"    Moved {script} output to {local_output}")
            else:
                print(f"    [WARNING] Expected output file {global_output} not found for {script}")
        
        return moved

    # Run WER scripts
    print(f"  Running WER scripts for {call_dir.name}...")
    for script_tuple in wer_scripts:
        script = script_tuple[0]
        args = script_tuple[1] if len(script_tuple) > 1 else []
        print(f"  Running {script}...")
        success = run_script_and_move_outputs(script, args, cwd=output_dir)
        if not success:
            print(f"    [ERROR] Failed to run {script}")

    # Run WER comparison
    print("  Running WER comparison...")
    try:
        result = subprocess.run([sys.executable, "compare_wer_metrics.py"],
                                cwd=PROJECT_ROOT / "wer",
                                check=True, capture_output=True, text=True, env=env)
        # Move the comparison CSV to call output dir
        wer_csv = PROJECT_ROOT / "wer" / "wer_metrics_comparison.csv"
        if wer_csv.exists():
            shutil.move(str(wer_csv), str(output_dir / "wer_metrics_comparison.csv"))
            print(f"    Moved WER comparison CSV to {output_dir / 'wer_metrics_comparison.csv'}")
        else:
            print(f"    [WARNING] Expected WER comparison CSV {wer_csv} not found")
    except subprocess.CalledProcessError as e:
        print(f"    [ERROR] WER comparison failed: {e.stderr}")

    # Run EER scripts
    print(f"  Running EER scripts for {call_dir.name}...")
    for script_tuple in eer_scripts:
        script = script_tuple[0]
        args = script_tuple[1] if len(script_tuple) > 1 else []
        print(f"  Running {script}...")
        success = run_script_and_move_outputs(script, args, cwd=output_dir)
        if not success:
            print(f"    [ERROR] Failed to run {script}")

    # Run EER comparison
    print("  Running EER comparison...")
    try:
        result = subprocess.run([sys.executable, "compare_eer_metrics.py"],
                                cwd=PROJECT_ROOT / "eer",
                                check=True, capture_output=True, text=True, env=env)
        eer_csv = PROJECT_ROOT / "eer" / "output" / "eer_metrics_comparison.csv"
        if eer_csv.exists():
            shutil.move(str(eer_csv), str(output_dir / "eer_metrics_comparison.csv"))
            print(f"    Moved EER comparison CSV to {output_dir / 'eer_metrics_comparison.csv'}")
        else:
            print(f"    [WARNING] Expected EER comparison CSV {eer_csv} not found")
    except subprocess.CalledProcessError as e:
        print(f"    [ERROR] EER comparison failed: {e.stderr}")

    # Check what files were actually created
    print(f"  Files in output directory after processing:")
    for file in output_dir.iterdir():
        print(f"    - {file.name}")


def aggregate_metrics_to_master_csv(base_dir, master_csv_path):
    """
    Aggregate all per-call wer_metrics_comparison.csv and eer_metrics_comparison.csv into a master CSV.
    Each row: call_id, WER/EER metrics from each call's output.
    """
    import csv
    from pathlib import Path

    base_dir = Path(base_dir)
    call_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    master_rows = []
    header = ["call_id"]
    first = True

    for call_dir in call_dirs:
        call_id = call_dir.name
        output_dir = call_dir / "output"
        wer_csv = output_dir / "wer_metrics_comparison.csv"
        eer_csv = output_dir / "eer_metrics_comparison.csv"
        row = {"call_id": call_id}

        # Read WER metrics
        if wer_csv.exists():
            with open(wer_csv, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for wer_row in reader:
                    for k, v in wer_row.items():
                        row[f"WER_{k}"] = v
                    break  # Only take the first row (or adjust as needed)
            if first:
                header += [f"WER_{k}" for k in wer_row.keys()]

        # Read EER metrics
        if eer_csv.exists():
            with open(eer_csv, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for eer_row in reader:
                    for k, v in eer_row.items():
                        row[f"EER_{k}"] = v
                    break  # Only take the first row (or adjust as needed)
            if first:
                header += [f"EER_{k}" for k in eer_row.keys()]

        master_rows.append(row)
        first = False

    # Write master CSV
    with open(master_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in master_rows:
            writer.writerow(row)
    print(f"[✔] Master metrics summary written to {master_csv_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Split annotations by call and download transcripts.")
    parser.add_argument("input_file", help="Path to combined annotations.json")
    parser.add_argument("--output_dir", default="calls",
                        help="Directory to write per-call folders and transcripts")
    args = parser.parse_args()

    split_annotations(args.input_file, args.output_dir)

    # After splitting, run WER/EER scripts for each call
    base_dir = Path(args.output_dir)
    for call_dir in base_dir.iterdir():
        if call_dir.is_dir():
            print(f"\n[INFO] Running WER/EER scripts for call: {call_dir.name}")
            try:
                run_wer_eer_scripts(call_dir)
            except Exception as e:
                print(f"[ERROR] Failed to run WER/EER scripts for {call_dir.name}: {e}")

    # Aggregate all per-call metrics to master CSV in project root
    aggregate_metrics_to_master_csv(base_dir, Path("master_metrics_summary.csv"))
