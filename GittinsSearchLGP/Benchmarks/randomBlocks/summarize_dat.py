#!/usr/bin/env python3
"""
Summarize a .dat CSV file: print mean ± std for every column.

Usage:
    python summarize_dat.py <path/to/file.dat>
"""

import sys
import csv
from pathlib import Path


def summarize(path: str) -> None:
    p = Path(path)
    if not p.exists():
        sys.exit(f"Error: file not found: {path}")

    rows = []
    with open(p, newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        if not headers:
            sys.exit("Error: no header row found.")
        for row in reader:
            rows.append(row)

    if not rows:
        sys.exit("Error: file has no data rows.")

    n = len(rows)
    print(f"File   : {p}")
    print(f"Rows   : {n}")
    print()

    # Compute mean and std for each column that is numeric
    col_width = max(len(h) for h in headers)
    header_line = f"{'column':<{col_width}}   {'mean':>14}   {'std':>14}   {'min':>14}   {'max':>14}"
    print(header_line)
    print("-" * len(header_line))

    for col in headers:
        values = []
        for row in rows:
            try:
                values.append(float(row[col]))
            except (ValueError, TypeError):
                pass  # skip non-numeric cells

        if not values:
            print(f"{col:<{col_width}}   {'(non-numeric)':>14}")
            continue

        m = sum(values) / len(values)
        variance = sum((v - m) ** 2 for v in values) / len(values)
        std = variance ** 0.5
        mn = min(values)
        mx = max(values)
        print(f"{col:<{col_width}}   {m:>14.6g}   {std:>14.6g}   {mn:>14.6g}   {mx:>14.6g}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"Usage: {sys.argv[0]} <path/to/file.dat>")
    summarize(sys.argv[1])
