#!/usr/bin/env bash
# Install vLLM into its own virtualenv, for racing against HIVE Limited.
#
# vLLM is a separate project under Apache-2.0. This script installs it FROM PyPI into an
# isolated venv; nothing from it is redistributed here.
#
# This is a large install -- several GB, including PyTorch and the CUDA runtime libraries.
# That size is itself informative: it is the stack HIVE Limited does not have.
set -euo pipefail

HOME_DIR="${HIVE_LIMITED_HOME:-$HOME/.hive-limited}"
ENGINES="${HIVE_LIMITED_ENGINES:-$HOME_DIR/engines}"
VENV="$ENGINES/vllm-env"

echo "==> installing vLLM into $VENV"
echo "    (several GB; PyTorch + CUDA runtime come with it)"
mkdir -p "$ENGINES"

command -v python3 >/dev/null || { echo "missing: python3"; exit 1; }
python3 -c 'import venv' 2>/dev/null || {
  echo "python3 venv module missing. On Debian/Ubuntu:  sudo apt install python3-venv"; exit 1; }

[ -d "$VENV" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip wheel >/dev/null
"$VENV/bin/pip" install vllm

"$VENV/bin/python" -c 'import vllm; print("vLLM", vllm.__version__, "OK")'

cat <<EOF

==> vLLM ready: $VENV

    Next: ./setup/download_models.sh   (it needs a HuggingFace-format model)

    If you run under WSL2 and vLLM fails on pinned memory, export:
        VLLM_WSL2_ENABLE_PIN_MEMORY=1
EOF
