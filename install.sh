#!/usr/bin/env bash
# HIVE Limited — one-command installer.
#
#   curl -fsSL https://raw.githubusercontent.com/Questeria/HIVE-Limited/main/install.sh | bash
#
# Written for someone who has never used a terminal before. Every check below explains, in
# plain language, what went wrong and exactly what to type to fix it — a beginner who hits
# "Permission denied" or "command not found" with no further explanation is simply stuck,
# and that is the moment most people give up on a project like this.
#
# It is safe to run more than once: anything already installed is detected and skipped.
# Everything lands in ONE folder (~/hive-limited by default) and nothing else is touched.

set -u

INSTALL_DIR="${HIVE_LIMITED_DIR:-$HOME/hive-limited}"
REPO="${HIVE_LIMITED_REPO:-Questeria/HIVE-Limited}"
BRANCH="${HIVE_LIMITED_BRANCH:-main}"
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3-1.7B}"

# ---- friendly output --------------------------------------------------------
if [ -t 1 ]; then
  B=$'\033[1m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; D=$'\033[2m'; N=$'\033[0m'
else
  B=""; G=""; Y=""; R=""; D=""; N=""
fi
step()  { printf '\n%s==>%s %s%s%s\n' "$G" "$N" "$B" "$*" "$N"; }
ok()    { printf '    %s✓%s %s\n' "$G" "$N" "$*"; }
info()  { printf '    %s\n' "$*"; }
warn()  { printf '    %s!%s %s\n' "$Y" "$N" "$*"; }

# A failure a beginner can act on: what is wrong, why it matters, what to type.
die() {
  printf '\n%s╭─ Stopped ─────────────────────────────────────────────%s\n' "$R" "$N"
  printf '%s│%s %s%s%s\n' "$R" "$N" "$B" "$1" "$N"
  shift
  printf '%s│%s\n' "$R" "$N"
  for line in "$@"; do printf '%s│%s %s\n' "$R" "$N" "$line"; done
  printf '%s╰───────────────────────────────────────────────────────%s\n\n' "$R" "$N"
  exit 1
}

cat <<EOF

  ${B}HIVE Limited${N}
  A small, fast AI text engine you can run on your own computer.
  ${D}This installer sets everything up for you. It takes about 10 minutes,
  most of which is downloading. You can leave it running.${N}
EOF

# ---- 1. is this the right kind of computer? ---------------------------------
step "Checking your computer"

case "$(uname -s 2>/dev/null || echo unknown)" in
  Linux) : ;;
  Darwin) die "This needs an NVIDIA graphics card, and Macs do not have one." \
    "HIVE Limited runs on NVIDIA GPUs only. There is no Mac version." \
    "" \
    "If you have a Windows PC with an NVIDIA card, you can use it there." ;;
  *) die "This installer needs Linux (or Windows with WSL)." \
    "You appear to be on: $(uname -s 2>/dev/null || echo 'an unknown system')" \
    "" \
    "On Windows, open PowerShell and run:    wsl --install" \
    "Then restart, open the 'Ubuntu' app, and run this installer again." ;;
esac

if grep -qi microsoft /proc/version 2>/dev/null; then
  ok "Windows with WSL — that works"
else
  ok "Linux"
fi

# ---- 2. graphics card -------------------------------------------------------
step "Checking your graphics card"

if [ ! -e /dev/nvidia0 ] && [ ! -e /dev/dxg ] && ! ls /dev/nvidia* >/dev/null 2>&1; then
  warn "Could not see an NVIDIA graphics card device."
  info "Continuing anyway — the check below is the one that really matters."
fi

DRIVER_LIB=""
for cand in /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/wsl/lib/libcuda.so.1 \
            /usr/lib64/libcuda.so.1 /usr/lib/libcuda.so.1; do
  [ -e "$cand" ] && { DRIVER_LIB="$cand"; break; }
done
if [ -z "$DRIVER_LIB" ] && command -v ldconfig >/dev/null 2>&1; then
  DRIVER_LIB=$(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1/{print $NF; exit}')
fi

if [ -z "$DRIVER_LIB" ]; then
  die "No NVIDIA graphics driver found." \
    "HIVE Limited needs an NVIDIA graphics card and its driver." \
    "It does NOT need the big 'CUDA Toolkit' — just the ordinary driver." \
    "" \
    "${B}On Windows + WSL:${N} install the normal NVIDIA driver in Windows" \
    "(nvidia.com/drivers), then close and reopen this window." \
    "" \
    "${B}On Ubuntu/Debian Linux:${N}" \
    "    sudo ubuntu-drivers autoinstall" \
    "    sudo reboot" \
    "" \
    "If you do not have an NVIDIA card, this software cannot run on this computer."
fi
ok "NVIDIA driver found"
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  [ -n "${GPU_NAME:-}" ] && ok "Graphics card: $GPU_NAME"
fi

# ---- 3. python --------------------------------------------------------------
step "Checking Python"

if ! command -v python3 >/dev/null 2>&1; then
  die "Python 3 is not installed." \
    "Python is the language HIVE Limited is written in." \
    "" \
    "${B}Ubuntu/Debian (including WSL):${N}" \
    "    sudo apt update && sudo apt install -y python3 python3-pip" \
    "" \
    "Then run this installer again."
fi
PY_VER=$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo "?")
case "$PY_VER" in
  3.9|3.1[0-9]) ok "Python $PY_VER" ;;
  *) warn "Python $PY_VER — 3.9 or newer is recommended; carrying on" ;;
esac

if ! python3 -c 'import numpy' >/dev/null 2>&1; then
  info "Installing numpy (the one library this needs)…"
  python3 -m pip install --user --quiet numpy 2>/dev/null \
    || python3 -m pip install --user --quiet --break-system-packages numpy 2>/dev/null \
    || die "Could not install numpy." \
        "numpy is the only extra library HIVE Limited needs." \
        "" \
        "Try installing it yourself with:" \
        "    sudo apt install -y python3-numpy" \
        "" \
        "Then run this installer again."
fi
ok "numpy ready"

# ---- 4. disk space ----------------------------------------------------------
step "Checking free space"
AVAIL_GB=$(df -Pk "$HOME" 2>/dev/null | awk 'NR==2{printf "%.0f", $4/1048576}')
if [ -n "${AVAIL_GB:-}" ] && [ "$AVAIL_GB" -lt 8 ] 2>/dev/null; then
  die "Not enough free space: about ${AVAIL_GB} GB available." \
    "The AI model alone is roughly 4 GB, so you need about 8 GB free." \
    "" \
    "Free some space and run this installer again."
fi
ok "${AVAIL_GB:-enough} GB free"

# ---- 5. get the software ----------------------------------------------------
step "Downloading HIVE Limited"

if [ -f "$INSTALL_DIR/serve.py" ]; then
  ok "Already downloaded — using $INSTALL_DIR"
else
  mkdir -p "$INSTALL_DIR" || die "Could not create $INSTALL_DIR" \
    "Check you have permission to write to your home folder."
  TARBALL="https://codeload.github.com/$REPO/tar.gz/refs/heads/$BRANCH"
  # Download to a file FIRST, then unpack. Piping curl straight into tar spills three
  # separate tool errors ("curl: (22)", "gzip: stdin: unexpected end of file", "tar: Child
  # returned status 1") above whatever friendly message follows, and that noise is exactly
  # what makes a beginner think they broke something.
  ARCHIVE="$INSTALL_DIR/.download.tar.gz"
  HTTP=000
  if command -v curl >/dev/null 2>&1; then
    HTTP=$(curl -sSL -w '%{http_code}' -o "$ARCHIVE" "$TARBALL" 2>/dev/null || echo 000)
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "$ARCHIVE" "$TARBALL" 2>/dev/null && HTTP=200 || HTTP=000
  else
    die "Neither curl nor wget is installed." \
      "One of these is needed to download files." \
      "" \
      "${B}Install one with:${N}" \
      "    sudo apt install -y curl" \
      "" \
      "Then run this installer again."
  fi

  case "$HTTP" in
    200) : ;;
    404) rm -f "$ARCHIVE"; die "Could not find the download." \
          "The server replied 'not found' for:" \
          "    $REPO (branch $BRANCH)" \
          "" \
          "This usually means the project is not published yet, or the link you" \
          "were given is out of date." \
          "" \
          "Please check for an updated link, or email ${B}ajdemarco10@gmail.com${N}." ;;
    000) rm -f "$ARCHIVE"; die "Could not reach the internet." \
          "The download could not connect at all." \
          "" \
          "• Check you are online." \
          "• If you are on a work or school network, it may be blocking GitHub." \
          "" \
          "Then run this installer again." ;;
    *)   rm -f "$ARCHIVE"; die "Download failed (server said: $HTTP)." \
          "Please try again in a few minutes — this is usually temporary." \
          "" \
          "If it keeps happening, email ${B}ajdemarco10@gmail.com${N} with that number." ;;
  esac

  if ! tar -xzf "$ARCHIVE" -C "$INSTALL_DIR" --strip-components=1 2>/dev/null; then
    rm -f "$ARCHIVE"
    die "The download arrived but could not be unpacked." \
      "The file was probably damaged in transit." \
      "" \
      "Run this installer again — it will download a fresh copy."
  fi
  rm -f "$ARCHIVE"
  ok "Downloaded to $INSTALL_DIR"
fi
chmod +x "$INSTALL_DIR"/*.sh "$INSTALL_DIR"/setup/*.sh 2>/dev/null

# ---- 6. the model -----------------------------------------------------------
step "Downloading the AI model (about 4 GB — this is the slow part)"

MODELS_DIR="${HIVE_LIMITED_MODELS:-$HOME/.hive-limited/models}"
MODEL_DIR="$MODELS_DIR/$(basename "$MODEL_REPO")"
if [ -f "$MODEL_DIR/config.json" ]; then
  ok "Model already downloaded"
else
  mkdir -p "$MODELS_DIR"
  export PATH="$HOME/.local/bin:$PATH"
  # `huggingface-cli` is DEPRECATED and no longer downloads — it prints a migration notice
  # and exits 0, which a caller checking only the exit code reads as success. `hf` replaced
  # it. Probe both so an older install still works, and prove the tool runs before handing
  # it a multi-GB job.
  HF=""
  for candidate in hf huggingface-cli; do
    command -v "$candidate" >/dev/null 2>&1 && { HF="$candidate"; break; }
  done
  if [ -z "$HF" ]; then
    info "Installing the downloader…"
    python3 -m pip install --user --quiet "huggingface_hub[cli]" 2>/dev/null \
      || python3 -m pip install --user --quiet --break-system-packages "huggingface_hub[cli]" 2>/dev/null
    for candidate in hf huggingface-cli; do
      command -v "$candidate" >/dev/null 2>&1 && { HF="$candidate"; break; }
    done
  fi
  [ -n "$HF" ] || die "Could not install the model downloader." \
    "Try:  python3 -m pip install --user 'huggingface_hub[cli]'" \
    "Then add this line to the end of your ~/.bashrc file:" \
    "    export PATH=\"\$HOME/.local/bin:\$PATH\""
  "$HF" --help >/dev/null 2>&1 || die "The downloader ($HF) is installed but does not run." \
    "Reinstall it with:" \
    "    python3 -m pip install --user --upgrade 'huggingface_hub[cli]'"
  info "Fetching $MODEL_REPO — leave this running, it can take several minutes."
  "$HF" download "$MODEL_REPO" --local-dir "$MODEL_DIR" \
    || die "Model download failed." \
        "This is usually a network problem. Run the installer again to resume —" \
        "already-downloaded pieces are kept, so it will pick up where it stopped."
  [ -f "$MODEL_DIR/config.json" ] || die "The download finished but the model is incomplete." \
    "Expected a config.json in:" \
    "    $MODEL_DIR" \
    "" \
    "Run the installer again — it resumes rather than starting over."
  ok "Model ready"
fi

# ---- 7. a launcher they can re-use ------------------------------------------
step "Setting up"

cat > "$INSTALL_DIR/start.sh" <<LAUNCHER
#!/usr/bin/env bash
# Start HIVE Limited. Created by the installer — just run it again any time.
cd "\$(dirname "\$0")"
export PATH="\$HOME/.local/bin:\$PATH"
MODEL="\${1:-$MODEL_DIR}"
PORT="\${2:-8080}"
echo ""
echo "  Starting HIVE Limited…"
echo "  When it says 'ready', open this in your web browser:"
echo ""
echo "      http://localhost:\$PORT"
echo ""
echo "  To stop it, press Ctrl and C together."
echo ""
exec python3 serve.py "\$MODEL" "\$PORT"
LAUNCHER
chmod +x "$INSTALL_DIR/start.sh"
ok "Created $INSTALL_DIR/start.sh"

cat <<EOF

${G}╭─ Done ────────────────────────────────────────────────${N}
${G}│${N} ${B}HIVE Limited is installed.${N}
${G}│${N}
${G}│${N} To start it, type this and press Enter:
${G}│${N}
${G}│${N}     ${B}$INSTALL_DIR/start.sh${N}
${G}│${N}
${G}│${N} Then open ${B}http://localhost:8080${N} in your web browser.
${G}│${N}
${G}│${N} ${D}Type a question, press LAUNCH, and watch it answer.${N}
${G}│${N} ${D}To race it against another AI engine, click ADD AN ENGINE.${N}
${G}╰───────────────────────────────────────────────────────${N}

EOF

printf 'Start it now? [Y/n] '
# The braces matter: a failed /dev/tty redirect prints bash's own error before the
# fallback runs unless the whole redirection is inside the silenced group.
{ read -r REPLY </dev/tty; } 2>/dev/null || REPLY="n"
case "${REPLY:-Y}" in
  ""|[Yy]*) exec "$INSTALL_DIR/start.sh" ;;
  *) echo "No problem — run $INSTALL_DIR/start.sh whenever you are ready."; echo ;;
esac
