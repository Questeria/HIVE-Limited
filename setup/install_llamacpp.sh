#!/usr/bin/env bash
# Install llama.cpp so you can race it against HIVE Limited.
#
# llama.cpp is the most popular way to run AI models on your own computer. It is a separate
# project under its own MIT licence: this script DOWNLOADS and BUILDS it on your machine and
# redistributes nothing.
#
# Everything lands in ~/.hive-limited/engines/ and nothing else is touched.
#
# Written to be readable by someone who has never built software before: every failure below
# says what went wrong and exactly what to type next, because "make: *** [all] Error 1" on
# its own is where most people stop.

set -u

HOME_DIR="${HIVE_LIMITED_HOME:-$HOME/.hive-limited}"
ENGINES="${HIVE_LIMITED_ENGINES:-$HOME_DIR/engines}"
SRC="$ENGINES/llama.cpp"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

if [ -t 1 ]; then
  B=$'\033[1m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; D=$'\033[2m'; N=$'\033[0m'
else
  B=""; G=""; Y=""; R=""; D=""; N=""
fi
step() { printf '\n%s==>%s %s%s%s\n' "$G" "$N" "$B" "$*" "$N"; }
ok()   { printf '    %s✓%s %s\n' "$G" "$N" "$*"; }
info() { printf '    %s\n' "$*"; }
die() {
  printf '\n%s╭─ Stopped ─────────────────────────────────────────────%s\n' "$R" "$N"
  printf '%s│%s %s%s%s\n' "$R" "$N" "$B" "$1" "$N"; shift
  printf '%s│%s\n' "$R" "$N"
  for l in "$@"; do printf '%s│%s %s\n' "$R" "$N" "$l"; done
  printf '%s╰───────────────────────────────────────────────────────%s\n\n' "$R" "$N"
  exit 1
}

cat <<EOF

  ${B}Installing llama.cpp${N}
  ${D}The most popular way to run AI models locally. Once it is installed you can
  race it against HIVE Limited and see both answer the same question.
  This builds it from source and takes about 5 minutes.${N}
EOF

step "Checking what you have"
for tool in git cmake; do
  command -v "$tool" >/dev/null 2>&1 || die "'$tool' is not installed." \
    "It is needed to download and build llama.cpp." \
    "" \
    "${B}Install it with:${N}" \
    "    sudo apt update && sudo apt install -y git cmake build-essential" \
    "" \
    "Then run this script again."
done
ok "git and cmake found"

if ! command -v nvcc >/dev/null 2>&1; then
  cat <<EOF

  ${Y}Note:${N} the CUDA Toolkit (nvcc) is not installed.

  llama.cpp needs it to use your graphics card. ${B}HIVE Limited does not${N} — that
  difference is the point of the comparison, not a gap in these instructions.

EOF
  # Ubuntu 22.04's packaged toolkit is CUDA 11.5, which current llama.cpp has outgrown.
  # Say so BEFORE a two-gigabyte download, not after a failed build.
  UBU=$(. /etc/os-release 2>/dev/null && echo "${VERSION_ID:-}")
  if [ "${UBU%%.*}" = "22" ]; then
    printf '  %s!%s Ubuntu 22.04: the packaged toolkit (CUDA 11.5) is too old for current\n' "$Y" "$N"
    printf '    llama.cpp. Use Ubuntu 24.04 (WSL: wsl --install -d Ubuntu-24.04), or install\n'
    printf '    a newer toolkit from NVIDIA: developer.nvidia.com/cuda-downloads\n\n'
  fi
  if [ "${LLAMA_CPU:-0}" != "1" ]; then
    # Driven from the web page there is nobody to answer a prompt, and a script waiting on
    # stdin forever is indistinguishable from a hung install. Fail with the reason instead.
    if [ "${HIVE_NONINTERACTIVE:-0}" = "1" ]; then
      die "The CUDA Toolkit is not installed." \
        "llama.cpp needs it to use your graphics card. HIVE Limited does not — that" \
        "difference is the point of the comparison." \
        "" \
        "Install it, then click Install again:" \
        "    sudo apt install -y nvidia-cuda-toolkit" \
        "" \
        "(It is a large download, around 2 GB.)"
    fi
    printf '  Install it now? This runs:  sudo apt install -y nvidia-cuda-toolkit\n'
    printf '  (about 2 GB; you will be asked for your password) [Y/n/cpu] '
    read -r REPLY </dev/tty 2>/dev/null || REPLY="n"
    case "${REPLY:-Y}" in
      ""|[Yy]*)
        sudo apt update && sudo apt install -y nvidia-cuda-toolkit \
          || die "The toolkit install failed." \
                 "Scroll up for the first error line. You can also try a CPU-only" \
                 "build to check the setup works:   LLAMA_CPU=1 $0"
        command -v nvcc >/dev/null 2>&1 \
          || die "Installed, but 'nvcc' is still not on PATH." \
                 "Close this terminal, open a new one, and run this script again." ;;
      [Cc]*) LLAMA_CPU=1
        info "CPU-only build: much slower, not a fair race — setup-check only." ;;
      *) die "Stopped so you can install the CUDA Toolkit first." \
             "Run:  sudo apt install -y nvidia-cuda-toolkit" \
             "Then run this script again." ;;
    esac
  fi
fi

step "Downloading llama.cpp"
mkdir -p "$ENGINES"
if [ -d "$SRC/.git" ]; then
  ok "Already downloaded — fetching any updates"
  git -C "$SRC" pull --ff-only >/dev/null 2>&1 || info "(could not update; building what is here)"
else
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$SRC" \
    || die "Download failed." \
        "Check your internet connection and try again." \
        "If you are on a company network, it may be blocking GitHub."
  ok "Downloaded"
fi

step "Building it (this is the slow part — about 5 minutes)"
CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=Release"
[ "${LLAMA_CPU:-0}" = "1" ] || CMAKE_FLAGS="$CMAKE_FLAGS -DGGML_CUDA=ON"
info "Using $JOBS processor cores. You can leave this running."

cmake -S "$SRC" -B "$SRC/build" $CMAKE_FLAGS >/dev/null 2>&1 \
  || die "Setting up the build failed." \
      "Run this to see the full error:" \
      "    cmake -S $SRC -B $SRC/build $CMAKE_FLAGS" \
      "" \
      "The usual cause is a missing compiler:" \
      "    sudo apt install -y build-essential"

cmake --build "$SRC/build" --config Release -j "$JOBS" --target llama-server \
  || die "The build failed." \
      "Scroll up for the first line containing the word 'error' — that is the real cause." \
      "" \
      "Two common ones:" \
      "  • ran out of memory   → try again with:  JOBS=2 $0" \
      "  • CUDA version clash  → try a CPU build:  LLAMA_CPU=1 $0" \
      "" \
      "Still stuck? Email ajdemarco10@gmail.com with that error line."

BIN="$SRC/build/bin/llama-server"
[ -x "$BIN" ] || die "The build finished but the program is missing." \
  "Expected it at: $BIN" \
  "Please email ajdemarco10@gmail.com — this one is on us, not you."

cat <<EOF

${G}╭─ Done ────────────────────────────────────────────────${N}
${G}│${N} ${B}llama.cpp is installed.${N}
${G}│${N}
${G}│${N} 1. Make sure you have the model file it needs:
${G}│${N}       ${B}./setup/download_models.sh${N}
${G}│${N} 2. Reload the HIVE Limited page in your browser
${G}│${N} 3. Pick ${B}llama.cpp${N} from the dropdown and press LAUNCH
${G}│${N}
${G}│${N} ${D}Both engines answer the same question, and the page shows${N}
${G}│${N} ${D}what it measured while you watched.${N}
${G}╰───────────────────────────────────────────────────────${N}

EOF
