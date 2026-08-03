#!/usr/bin/env bash
# HIVE Limited — doctor. Run this when something is wrong and it will say what.
#
#   ./doctor.sh
#
# Every line below is one check: what passed, or what failed AND exactly what to type to
# fix it. Copy-paste the whole output into an email to ajdemarco10@gmail.com and it is a
# complete bug report — that is the other half of why this file exists.

set -u
cd "$(dirname "$0")"

if [ -t 1 ]; then
  B=$'\033[1m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; D=$'\033[2m'; N=$'\033[0m'
else
  B=""; G=""; Y=""; R=""; D=""; N=""
fi
FAILS=0
pass() { printf '  %s✓%s %s\n' "$G" "$N" "$*"; }
warn() { printf '  %s!%s %s\n' "$Y" "$N" "$*"; }
fail() { printf '  %s✗ %s%s\n' "$R" "$*" "$N"; FAILS=$((FAILS+1)); }
fix()  { printf '      %s→ %s%s\n' "$D" "$*" "$N"; }

printf '\n%sHIVE Limited doctor%s\n\n' "$B" "$N"

# ---- 1. system --------------------------------------------------------------
case "$(uname -s 2>/dev/null || echo unknown)" in
  Linux)
    if grep -qi microsoft /proc/version 2>/dev/null; then pass "Windows with WSL"
    else pass "Linux"; fi ;;
  *) fail "Not Linux — this build runs on Linux or Windows-with-WSL."
     fix "On Windows: open PowerShell as administrator, run:  wsl --install" ;;
esac

# ---- 2. NVIDIA driver -------------------------------------------------------
DRIVER_LIB=""
for cand in /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/wsl/lib/libcuda.so.1 \
            /usr/lib64/libcuda.so.1 /usr/lib/libcuda.so.1; do
  [ -e "$cand" ] && { DRIVER_LIB="$cand"; break; }
done
[ -z "$DRIVER_LIB" ] && command -v ldconfig >/dev/null 2>&1 && \
  DRIVER_LIB=$(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1/{print $NF; exit}')
if [ -n "$DRIVER_LIB" ]; then
  pass "NVIDIA driver library: $DRIVER_LIB"
else
  fail "No NVIDIA driver library (libcuda.so.1) found."
  fix "WSL: install the normal Windows driver from nvidia.com/drivers, then reopen this window."
  fix "Ubuntu: sudo ubuntu-drivers autoinstall && sudo reboot"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  [ -n "${GPU:-}" ] && pass "Graphics card: $GPU" || warn "nvidia-smi present but reported no GPU"
else
  warn "nvidia-smi not found (not fatal — the library check above is the real one)"
fi

# ---- 3. python + numpy ------------------------------------------------------
if command -v python3 >/dev/null 2>&1; then
  pass "python3 $(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null)"
  if python3 -c 'import numpy' >/dev/null 2>&1; then pass "numpy installed"
  else fail "numpy is missing."
       fix "python3 -m pip install --user numpy    (or: sudo apt install -y python3-numpy)"; fi
else
  fail "python3 is not installed."
  fix "sudo apt update && sudo apt install -y python3 python3-pip"
fi

# ---- 4. the shipped kernels are intact --------------------------------------
if python3 - <<'EOF' 2>/dev/null
import hashlib, json, sys
from pathlib import Path
m = json.loads(Path("kernels/MANIFEST.json").read_text())
bad = [k["file"] for k in m["kernels"]
       if hashlib.sha256((Path("kernels")/k["file"]).read_bytes()).hexdigest() != k["sha256"]]
sys.exit(1 if bad else 0)
EOF
then
  pass "kernel files match their sha256 manifest"
else
  fail "A kernel file is missing or does not match kernels/MANIFEST.json."
  fix "Re-download HIVE Limited — the copy on disk is damaged:"
  fix "curl -fsSL https://raw.githubusercontent.com/Questeria/HIVE-Limited/main/install.sh | bash"
fi

# ---- 5. the model -----------------------------------------------------------
MODELS="${HIVE_LIMITED_MODELS:-${HIVE_LIMITED_HOME:-$HOME/.hive-limited}/models}"
FOUND_MODEL=""
for d in "$MODELS"/*/; do
  [ -f "$d/config.json" ] && [ -f "$d/tokenizer.json" ] && \
    ls "$d"/*.safetensors >/dev/null 2>&1 && { FOUND_MODEL="$d"; break; }
done
if [ -n "$FOUND_MODEL" ]; then
  pass "model ready: $(basename "$FOUND_MODEL")"
  if ls "$MODELS"/*.gguf >/dev/null 2>&1 || ls "$FOUND_MODEL"/*.gguf >/dev/null 2>&1; then
    pass "GGUF file present (llama.cpp races possible)"
  else
    warn "no GGUF file — llama.cpp races need one:  ./setup/download_models.sh"
  fi
else
  fail "No complete model under $MODELS"
  fix "./setup/download_models.sh    (or re-run install.sh, which downloads one)"
fi

# ---- 6. free space + port ---------------------------------------------------
AVAIL_GB=$(df -Pk "$HOME" 2>/dev/null | awk 'NR==2{printf "%.0f", $4/1048576}')
if [ -n "${AVAIL_GB:-}" ] && [ "$AVAIL_GB" -lt 2 ] 2>/dev/null; then
  warn "only ${AVAIL_GB} GB free — downloads may fail"
else
  pass "${AVAIL_GB:-?} GB free disk space"
fi
PORT="${1:-8080}"
if command -v python3 >/dev/null 2>&1 && python3 - "$PORT" <<'EOF' 2>/dev/null
import socket, sys
s = socket.socket()
try:
    s.bind(("127.0.0.1", int(sys.argv[1]))); s.close(); sys.exit(0)
except OSError:
    sys.exit(1)
EOF
then
  pass "port $PORT is free"
else
  warn "port $PORT is in use — HIVE Limited may already be running (that is fine),"
  fix "or start on another port:  ./start.sh <model_dir> 8081"
fi

# ---- 7. rivals (informational) ----------------------------------------------
ENGINES="${HIVE_LIMITED_ENGINES:-${HIVE_LIMITED_HOME:-$HOME/.hive-limited}/engines}"
[ -x "$ENGINES/llama.cpp/build/bin/llama-server" ] || command -v llama-server >/dev/null 2>&1 \
  && pass "rival llama.cpp installed" \
  || warn "rival llama.cpp not installed (optional):  ./setup/install_llamacpp.sh"
[ -x "$ENGINES/vllm-env/bin/python" ] \
  && pass "rival vLLM installed" \
  || warn "rival vLLM not installed (optional):  ./setup/install_vllm.sh"

# ---- verdict ---------------------------------------------------------------
echo
if [ "$FAILS" = 0 ]; then
  if [ -x ./start.sh ]; then START="./start.sh"
  else START="python3 serve.py ${FOUND_MODEL:-<model_dir>}"; fi
  printf '%sEverything needed is in place.%s Start it with:  %s%s%s\n\n' "$G$B" "$N" "$B" "$START" "$N"
  exit 0
else
  printf '%s%d problem(s) above%s — each has its fix on the line beneath it.\n' "$R$B" "$FAILS" "$N"
  printf 'Still stuck? Email %sajdemarco10@gmail.com%s with everything this printed.\n\n' "$B" "$N"
  exit 1
fi
