#!/bin/bash
# Script all models in all_datasize_models into all_datasize_models_scripted,
# preserving the folder structure:
#
#   all_datasize_models/
#     datasize_{p}/
#       seed_{i}/
#         best_model_*.pt   ← raw checkpoints
#
#   all_datasize_models_scripted/
#     datasize_{p}/
#       seed_{i}/
#         model_*.pt        ← TorchScript versions
#
# Usage:
#   ./script_datasize_models.sh [src_root] [dst_root] [datasize_filter]
#
# Defaults:
#   src_root       = ../all_datasize_models   (relative to this script)
#   dst_root       = ../all_datasize_models_scripted
#   datasize_filter = (empty = process all datasize_* folders)
#
# Examples:
#   ./script_datasize_models.sh                          # all datasizes
#   ./script_datasize_models.sh "" "" datasize_1         # only datasize_1
#   ./script_datasize_models.sh /my/src /my/dst datasize_0.4

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/script_local_models.py"

# Resolve source and destination roots
SRC="${1:-$SCRIPT_DIR/../all_datasize_models}"
DST="${2:-$SCRIPT_DIR/../all_datasize_models_scripted}"
FILTER="${3:-}"   # optional datasize folder name, e.g. "datasize_1"
SRC="$(realpath "$SRC")"
DST="$(realpath -m "$DST")"

# Find the Python interpreter (prefer the project venv)
VENV_PYTHON="$(realpath -m "$SCRIPT_DIR/../../.." )/.venv/bin/python"
if [ -x "$VENV_PYTHON" ]; then
  PYTHON="$VENV_PYTHON"
elif command -v python3 &>/dev/null; then
  PYTHON="python3"
else
  PYTHON="python"
fi

echo ""
echo "=========================================="
echo "  Scripting all datasize models"
echo "=========================================="
echo "  Source : $SRC"
echo "  Dest   : $DST"
echo "  Filter : ${FILTER:-(all)}"
echo "  Python : $PYTHON"
echo ""

if [ ! -d "$SRC" ]; then
  echo "ERROR: Source directory not found: $SRC"
  exit 1
fi

total=0
success=0
failed=0
failed_list=()

# Iterate datasize_* / seed_* hierarchy
for datasize_dir in "$SRC"/datasize_*/; do
  [ -d "$datasize_dir" ] || continue
  datasize=$(basename "$datasize_dir")

  # Skip if a filter is set and this folder doesn't match
  if [ -n "$FILTER" ] && [ "$datasize" != "$FILTER" ]; then
    continue
  fi

  for seed_dir in "$datasize_dir"seed_*/; do
    [ -d "$seed_dir" ] || continue
    seed=$(basename "$seed_dir")

    # Count .pt files to process
    n_pt=$(find "$seed_dir" -maxdepth 1 -name "*.pt" | wc -l)
    if [ "$n_pt" -eq 0 ]; then
      echo "  [SKIP] $datasize/$seed — no .pt files found"
      continue
    fi

    out_dir="$DST/$datasize/$seed"
    mkdir -p "$out_dir"

    echo "-------------------------------------------"
    echo "  $datasize/$seed  ($n_pt model(s))  →  $out_dir"
    echo "-------------------------------------------"

    $PYTHON "$PYTHON_SCRIPT" \
      --artifacts-dir "$seed_dir" \
      --output-dir    "$out_dir"

    exit_code=$?
    total=$((total + n_pt))
    if [ $exit_code -eq 0 ]; then
      success=$((success + n_pt))
    else
      failed=$((failed + n_pt))
      failed_list+=("$datasize/$seed")
      echo "  WARNING: scripting failed for $datasize/$seed (exit $exit_code)"
    fi
  done
done

echo ""
echo "=========================================="
echo "  DONE"
echo "=========================================="
echo "  Models processed : $total"
echo "  Successful       : $success"
echo "  Failed           : $failed"
if [ ${#failed_list[@]} -gt 0 ]; then
  echo ""
  echo "  Failed folders:"
  for f in "${failed_list[@]}"; do
    echo "    - $f"
  done
fi
echo ""
echo "  Scripted models saved to: $DST"
echo ""
