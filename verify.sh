#!/usr/bin/env bash
# Reproducibility check: determinism + agreement with an independent numpy reference.
set -euo pipefail
MODEL="${1:-${HIVE_LIMITED_MODEL:-}}"
if [ -z "$MODEL" ]; then
  echo "usage: ./verify.sh <model_dir>   (or set HIVE_LIMITED_MODEL)" >&2
  exit 2
fi
exec python3 "$(dirname "$0")/verify.py" "$MODEL" "${2:-6}"
