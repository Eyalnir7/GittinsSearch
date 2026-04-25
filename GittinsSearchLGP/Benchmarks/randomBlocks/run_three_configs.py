#!/usr/bin/env python3
import argparse
import subprocess
import threading
import shutil
from pathlib import Path
import re
import sys

CONFIGS = [
    {
        "name": "first",
        "numObjLowerBound": 4,
        "numObjUpperBound": 4,
        "numGoalsUpperBound": 4,
        "numBlockedGoalsUpperBound": 1,
    },
    {
        "name": "second",
        "numObjLowerBound": 3,
        "numObjUpperBound": 3,
        "numGoalsUpperBound": 3,
        "numBlockedGoalsUpperBound": 2,
    },
    {
        "name": "third",
        "numObjLowerBound": 2,
        "numObjUpperBound": 2,
        "numGoalsUpperBound": 2,
        "numBlockedGoalsUpperBound": 2,
    },
]

def update_cfg(cfg_path: Path, updates: dict):
    lines = cfg_path.read_text(encoding="utf-8").splitlines()
    out = []
    seen = set()

    for line in lines:
        replaced = False
        for k, v in updates.items():
            if re.match(rf"\s*{re.escape(k)}\s*:", line):
                out.append(f"{k}: {v}")
                seen.add(k)
                replaced = True
                break
        if not replaced:
            out.append(line)

    for k, v in updates.items():
        if k not in seen:
            out.append(f"{k}: {v}")

    cfg_path.write_text("\n".join(out) + "\n", encoding="utf-8")

def run_one(name, cfg_updates, exe_path: Path, cfg_path: Path, logs_dir: Path, work_dir: Path):
    log_out = logs_dir / f"{name}.out.log"
    log_err = logs_dir / f"{name}.err.log"

    # WARNING: This edits a shared file. Parallel runs are only safe
    # if x.exe reads rai.cfg once at startup and never rereads it.
    update_cfg(cfg_path, cfg_updates)

    with log_out.open("w") as out, log_err.open("w") as err:
        p = subprocess.Popen([str(exe_path)], cwd=str(work_dir), stdout=out, stderr=err)
        return p.wait()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataPercentage", type=float)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    exe_path = script_dir / "x.exe"
    cfg_path = script_dir / "rai.cfg"
    logs_dir = script_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    if not exe_path.exists():
        print(f"ERROR: not found: {exe_path}", file=sys.stderr)
        sys.exit(1)
    if not cfg_path.exists():
        print(f"ERROR: not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    backup_cfg = script_dir / "rai.cfg.bak"
    shutil.copy2(cfg_path, backup_cfg)

    threads = []
    results = {}

    def worker(cfg):
        updates = {
            "numObjLowerBound": cfg["numObjLowerBound"],
            "numObjUpperBound": cfg["numObjUpperBound"],
            "numGoalsUpperBound": cfg["numGoalsUpperBound"],
            "numBlockedGoalsUpperBound": cfg["numBlockedGoalsUpperBound"],
            "dataPercentage": args.dataPercentage,
        }
        name = cfg["name"]
        rc = run_one(name, updates, exe_path, cfg_path, logs_dir, script_dir)
        results[name] = rc

    try:
        for cfg in CONFIGS:
            t = threading.Thread(target=worker, args=(cfg,), daemon=False)
            t.start()
            threads.append(t)

        for t in threads:
            t.join()
    finally:
        shutil.copy2(backup_cfg, cfg_path)
        backup_cfg.unlink(missing_ok=True)

    print("\n=== Run summary ===")
    for name in ["first", "second", "third"]:
        rc = results.get(name, "N/A")
        print(f"{name}: {rc} (logs: {logs_dir / (name + '.out.log')}, {logs_dir / (name + '.err.log')})")

    # nonzero exit if any failed
    if any(isinstance(results.get(n), int) and results[n] != 0 for n in results):
        sys.exit(1)

if __name__ == "__main__":
    main()
