#!/usr/bin/env python3
"""
Grid-search selected parameters in rai.cfg and run ./x.exe for each combo.

Edits ONLY these keys (everything else stays byte-for-byte the same except the
numeric value on those lines):
- solver (set to GITTINS)
- numTaskPlans
- GITTINS/numWaypoints

Usage:
  python3 grid_search_Gittins.py
"""

import itertools
import os
import re
import subprocess
from datetime import datetime

CFG_PATH = "rai.cfg"
EXECUTABLE = "./x.exe"

GRID = {
    "numTaskPlans": [5, 10, 25],
    "GITTINS/numWaypoints": [20, 50, 100],
}

# Keys we will edit in the file
EDIT_KEYS = [
    "solver",
    "numTaskPlans",
    "GITTINS/numWaypoints",
]

# Format values appropriately
def format_value(key: str, v) -> str:
    if key == "solver":
        return "GITTINS"
    return str(v)


def load_cfg_lines(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()


def write_cfg_lines(path: str, lines: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def replace_key_value_in_lines(lines: list[str], key: str, new_value: str) -> tuple[list[str], bool]:
    """
    Replace only the value part on the line that starts with: optional ws + key + optional ws + ":".
    Preserves comments and all other text on the line.
    """
    # Capture:
    #  (prefix)   key: <spaces>
    #  (value)    anything up to optional comment
    #  (suffix)   optional spaces + optional comment + newline
    pattern = re.compile(rf"^(\s*{re.escape(key)}\s*:\s*)([^#\n]*?)(\s*(#.*)?\n?)$")

    replaced = False
    out = []
    for line in lines:
        m = pattern.match(line)
        if m and not replaced:
            prefix, _old_value, suffix = m.group(1), m.group(2), m.group(3)
            out.append(prefix + new_value + suffix)
            replaced = True
        else:
            out.append(line)
    return out, replaced


def ensure_all_keys_present(original_lines: list[str]) -> None:
    missing = []
    for key in EDIT_KEYS:
        pat = re.compile(rf"^\s*{re.escape(key)}\s*:\s*")
        if not any(pat.match(ln) for ln in original_lines):
            missing.append(key)
    if missing:
        raise RuntimeError(f"Missing required keys in {CFG_PATH}: {missing}")


def run_once(run_dir: str) -> int:
    os.makedirs(run_dir, exist_ok=True)
    err_path = os.path.join(run_dir, "stderr.txt")

    with open(err_path, "w", encoding="utf-8") as err_f:
        p = subprocess.Popen([EXECUTABLE], stdout=None, stderr=subprocess.PIPE, text=True)

        # Stream stderr live and save to file
        for line in p.stderr:
            print(line, end="")
            err_f.write(line)

        p.wait()
        return p.returncode


def main():
    original_lines = load_cfg_lines(CFG_PATH)
    ensure_all_keys_present(original_lines)

    # Make a backup once
    backup_path = CFG_PATH + ".bak"
    if not os.path.exists(backup_path):
        write_cfg_lines(backup_path, original_lines)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = f"grid_search_results/gridGittins_results_{stamp}"
    os.makedirs(results_root, exist_ok=True)

    combos = itertools.product(
        GRID["numTaskPlans"],
        GRID["GITTINS/numWaypoints"],
    )

    idx = 0
    for numTaskPlans, numWaypoints in combos:
        idx += 1
        params = {
            "solver": "GITTINS",
            "numTaskPlans": numTaskPlans,
            "GITTINS/numWaypoints": numWaypoints,
        }

        # Start from pristine original every time
        lines = list(original_lines)

        # Apply edits (only these lines change)
        for key in EDIT_KEYS:
            lines, ok = replace_key_value_in_lines(lines, key, format_value(key, params[key]))
            if not ok:
                raise RuntimeError(f"Failed to replace key {key} in {CFG_PATH}")

        # Write rai.cfg
        write_cfg_lines(CFG_PATH, lines)

        # Run and log
        run_name = (
            f"{idx:03d}_"
            f"numTaskPlans={numTaskPlans}_"
            f"numWaypoints={numWaypoints}"
        )
        run_dir = os.path.join(results_root, run_name)

        # Save the exact config used for this run
        os.makedirs(run_dir, exist_ok=True)
        write_cfg_lines(os.path.join(run_dir, "rai.cfg.used"), lines)

        print(f"\n{'='*60}")
        print(f"Running combo {idx}: numTaskPlans={numTaskPlans}, numWaypoints={numWaypoints}")
        print(f"{'='*60}\n")
        
        code = run_once(run_dir)

    # Restore original cfg at end (optional but usually desired)
    write_cfg_lines(CFG_PATH, original_lines)
    print(f"\nDone. Results in: {results_root}")
    print(f"Original cfg backed up at: {backup_path}")


if __name__ == "__main__":
    main()
