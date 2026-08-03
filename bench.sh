#!/usr/bin/env bash
# Measure HIVE Limited on this machine. Prints only what it just measured.
set -euo pipefail
MODEL="${1:-${HIVE_LIMITED_MODEL:-}}"
if [ -z "$MODEL" ]; then
  echo "usage: ./bench.sh <model_dir>    (or set HIVE_LIMITED_MODEL)" >&2
  exit 2
fi
exec python3 "$(dirname "$0")/bench.py" "$MODEL" "${2:-64}"
