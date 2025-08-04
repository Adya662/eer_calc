#!/usr/bin/env python3
import os, sys, time, shutil, subprocess
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent / ".env"
if env_path.exists(): load_dotenv(env_path)

def main(call_dir: str):
    start = time.time()
    cd = Path(call_dir)
    out = cd/"output"; out.mkdir(exist_ok=True)
    for name in ("ref_transcript.json","gt_transcript.json"):
        shutil.copy(cd/name, out/name)

    script = Path(__file__).parent.parent/"eer"/"eer_stanford.py"
    cmd = [sys.executable, str(script)]
    subprocess.run(cmd, cwd=out, check=True, env=os.environ)

    print(f"[{cd.name}] eer_stanford done in {(time.time()-start):.2f}s")

if __name__=="__main__":
    if len(sys.argv)!=2:
        print("Usage: run_eer_stanford.py <call_dir>"); sys.exit(1)
    main(sys.argv[1])