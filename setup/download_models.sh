#!/usr/bin/env bash
# Download an AI model for HIVE Limited — and, so the race means something, the matching
# file for the rival engine too.
#
#   ./setup/download_models.sh              pick from a menu
#   ./setup/download_models.sh qwen3-4b     go straight to one
#   ./setup/download_models.sh Qwen/Qwen3-8B   any Qwen3 model from HuggingFace
#
# WHY TWO FILES PER MODEL: HIVE Limited and vLLM read HuggingFace files; llama.cpp reads a
# different format called GGUF. They are the same AI, packaged differently. Fetching both
# is what lets the two engines answer the same question with the same model — which is the
# only way the comparison means anything.
#
# Models are downloaded from HuggingFace under THEIR OWN licences (Qwen3 is Apache-2.0).
# Nothing is redistributed by this repository. Everything lands in ~/.hive-limited/models.

set -u

MODELS="${HIVE_LIMITED_MODELS:-${HIVE_LIMITED_HOME:-$HOME/.hive-limited}/models}"

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

#  key | display name | HF repo | GGUF repo | GGUF file | size | note
#
# The three larger sizes are the models our published comparisons were measured on, so
# anyone reading a row on the arena board can install the exact model behind it and run the
# same pairing on their own machine. That is NOT the same as reproducing our numbers —
# those came from the full engine, which is not in this repository.
CATALOGUE="
qwen3-0.6b|Qwen3-0.6B|Qwen/Qwen3-0.6B|Qwen/Qwen3-0.6B-GGUF|Qwen3-0.6B-Q4_K_M.gguf|~1.5 GB|smallest and fastest — good first run, or a modest card
qwen3-1.7b|Qwen3-1.7B|Qwen/Qwen3-1.7B|Qwen/Qwen3-1.7B-GGUF|Qwen3-1.7B-Q4_K_M.gguf|~4 GB|the default, 8 GB card — the model behind our H100 vs TensorRT-LLM rows
qwen3-4b|Qwen3-4B|Qwen/Qwen3-4B|Qwen/Qwen3-4B-GGUF|Qwen3-4B-Q4_K_M.gguf|~9 GB|12 GB card — the model behind our vLLM rows
qwen3-8b|Qwen3-8B|Qwen/Qwen3-8B|Qwen/Qwen3-8B-GGUF|Qwen3-8B-Q4_K_M.gguf|~17 GB|the model behind our llama.cpp rows — big, and wants ~20 GB RAM to load
"

row_for() { printf '%s\n' "$CATALOGUE" | awk -F'|' -v k="$1" '$1==k{print;exit}'; }

# ---- choose ------------------------------------------------------------------
CHOICE="${1:-}"

if [ -z "$CHOICE" ] && [ "${HIVE_NONINTERACTIVE:-0}" = "1" ]; then
  die "No model was named." \
    "When run from the web page a model must be given, because there is nobody" \
    "here to answer a menu." \
    "" \
    "From a terminal, run it without arguments to get the menu:" \
    "    ./setup/download_models.sh"
fi

if [ -z "$CHOICE" ]; then
  cat <<EOF

  ${B}Which AI model would you like?${N}
  ${D}All of these work with HIVE Limited. Bigger models give better answers
  and need more graphics memory.${N}

EOF
  i=0
  printf '%s\n' "$CATALOGUE" | while IFS='|' read -r k name hf gr gf size note; do
    [ -z "$k" ] && continue
    i=$((i+1))
    mark="  "
    [ -d "$MODELS/$name" ] && mark="${G}✓${N} "
    printf '   %s%s) %s%-12s%s %-8s %s%s%s\n' "$mark" "$i" "$B" "$name" "$N" "$size" "$D" "$note" "$N"
  done
  printf '\n   %sA ✓ means you already have it.%s\n' "$D" "$N"
  printf '   %s2, 3 and 4 are the models our published comparisons used.%s\n' "$D" "$N"
  printf '\n  Number (or Enter for the default, Qwen3-1.7B): '
  read -r PICK </dev/tty 2>/dev/null || PICK=""
  case "${PICK:-2}" in
    1) CHOICE="qwen3-0.6b" ;;
    2|"") CHOICE="qwen3-1.7b" ;;
    3) CHOICE="qwen3-4b" ;;
    4) CHOICE="qwen3-8b" ;;
    *) die "'$PICK' is not one of the choices." "Run the script again and enter 1, 2, 3 or 4." ;;
  esac
fi

ROW=$(row_for "$CHOICE")
if [ -n "$ROW" ]; then
  IFS='|' read -r KEY NAME REPO_HF GGUF_REPO GGUF_FILE SIZE NOTE <<EOF
$ROW
EOF
else
  # Not in the menu: treat it as a HuggingFace repo id, e.g. Qwen/Qwen3-8B
  case "$CHOICE" in
    */*) REPO_HF="$CHOICE"; NAME="$(basename "$CHOICE")"; GGUF_REPO="${CHOICE}-GGUF"
         GGUF_FILE=""; SIZE="unknown"; NOTE="from HuggingFace" ;;
    *) die "'$CHOICE' is not a model I know." \
        "Run without arguments to pick from the menu:" \
        "    ./setup/download_models.sh" \
        "" \
        "Or give a full HuggingFace name, for example:" \
        "    ./setup/download_models.sh Qwen/Qwen3-8B" ;;
  esac
  printf '\n  %sNote:%s only Qwen3 models work — this build ships kernels for that\n' "$Y" "$N"
  printf '  architecture only. Others will download but will not load.\n'
fi

DEST="$MODELS/$NAME"
mkdir -p "$MODELS"

# ---- downloader --------------------------------------------------------------
# HuggingFace renamed the tool: `huggingface-cli` is deprecated and NO LONGER WORKS — it
# now just prints a migration notice and exits 0, which is worse than failing, because a
# caller that only checks the exit code believes the download succeeded. `hf` is the
# replacement. Both names are probed so an older install still works.
export PATH="$HOME/.local/bin:$PATH"
HF=""
for candidate in hf huggingface-cli; do
  command -v "$candidate" >/dev/null 2>&1 && { HF="$candidate"; break; }
done
if [ -z "$HF" ]; then
  step "Installing the downloader"
  python3 -m pip install --user --quiet "huggingface_hub[cli]" 2>/dev/null \
    || python3 -m pip install --user --quiet --break-system-packages "huggingface_hub[cli]" 2>/dev/null
  for candidate in hf huggingface-cli; do
    command -v "$candidate" >/dev/null 2>&1 && { HF="$candidate"; break; }
  done
fi
[ -n "$HF" ] || die "Could not install the model downloader." \
  "Try this yourself:" \
  "    python3 -m pip install --user 'huggingface_hub[cli]'" \
  "" \
  "Then add this to the end of your ~/.bashrc file and reopen the terminal:" \
  "    export PATH=\"\$HOME/.local/bin:\$PATH\""

# Prove the tool actually runs before trusting it with a multi-GB download.
if ! "$HF" --help >/dev/null 2>&1; then
  die "The downloader ($HF) is installed but does not run." \
    "Reinstall it with:" \
    "    python3 -m pip install --user --upgrade 'huggingface_hub[cli]'"
fi

# ---- 1/2 the HuggingFace copy (HIVE Limited + vLLM) --------------------------
step "Downloading $NAME for HIVE Limited  ($SIZE)"
if [ -f "$DEST/config.json" ]; then
  ok "already downloaded"
else
  info "Leave this running — it can take several minutes."
  "$HF" download "$REPO_HF" --local-dir "$DEST" \
    || die "Download failed." \
        "This is almost always a network problem." \
        "" \
        "Run the same command again — finished pieces are kept, so it carries on" \
        "from where it stopped rather than starting over."
  ok "$NAME ready"
fi

# ---- 2/2 the GGUF copy (llama.cpp) -------------------------------------------
step "Downloading the llama.cpp copy of the same model"
if ls "$MODELS"/*"$(printf '%s' "$NAME")"*.gguf >/dev/null 2>&1; then
  ok "already downloaded"
elif [ -z "$GGUF_FILE" ]; then
  info "No standard GGUF filename known for this model — skipping."
  info "HIVE Limited will still run it; llama.cpp will not be able to race it."
else
  # DO NOT trust a hardcoded filename. Two of the four in the original catalogue did not
  # exist upstream, and the download failed with the script still reporting success. Ask
  # the repository what it actually has, and prefer the smallest-bit build available:
  # llama.cpp's Q4_K_M (~5 bpw) is the closest match to what HIVE reads, so it is the only
  # choice that makes the race about the engine rather than the format.
  info "Asking the repository which files it has…"
  PICKED=$(python3 - "$GGUF_REPO" <<'PY' 2>/dev/null
import sys
try:
    from huggingface_hub import list_repo_files
    files = [f for f in list_repo_files(sys.argv[1]) if f.lower().endswith(".gguf")]
except Exception:
    sys.exit(0)
# Preference order: closest bit-width to HIVE's 5.00 bpw first.
for want in ("q4_k_m", "q4_k_s", "q4_0", "q5_k_m", "q5_0", "q6_k", "q8_0"):
    for f in files:
        if want in f.lower():
            print(f); sys.exit(0)
if files:
    print(files[0])
PY
  )

  if [ -z "$PICKED" ]; then
    GGUF_FAILED=1
    GGUF_WHY="could not reach the repository, or it has no .gguf files"
  else
    info "Fetching $PICKED"
    if "$HF" download "$GGUF_REPO" "$PICKED" --local-dir "$MODELS" >/dev/null 2>&1; then
      : # fall through to the artifact check below
    fi
    # THE FILE IS THE TRUTH, not the exit code. An earlier version piped the download
    # through `tail`, so the `if` tested tail's status — always success — and reported a
    # missing file as installed. Check for the artifact instead.
    if [ -f "$MODELS/$PICKED" ]; then
      ok "llama.cpp copy ready: $PICKED"
      case "$(printf '%s' "$PICKED" | tr 'A-Z' 'a-z')" in
        *q4*) : ;;
        *) printf '\n  %s!%s Note: the only build published for this model is %s.\n' "$Y" "$N" "$PICKED"
           printf '  That is a LARGER format than HIVE reads, so a race against it is not\n'
           printf '  a like-for-like comparison — part of any speed difference will be the\n'
           printf '  file size, not the engine. The arena says so on the results page.\n' ;;
      esac
    else
      GGUF_FAILED=1
      GGUF_WHY="the download did not produce $PICKED"
    fi
  fi
fi

# ---- done --------------------------------------------------------------------
# HALF A JOB IS NOT A SUCCESS. This used to print "is installed" and exit 0 even when the
# GGUF fetch had failed — so the web button turned into "✓ Installed", and the user was
# left with a model llama.cpp still could not read and no indication why. The exit code is
# what the button believes, so it has to tell the truth.
if [ "${GGUF_FAILED:-0}" = "1" ]; then
  cat <<EOF

${Y}╭─ Partly done ─────────────────────────────────────────${N}
${Y}│${N} ${B}$NAME is installed and HIVE Limited can run it.${N}
${Y}│${N}
${Y}│${N} The llama.cpp copy could NOT be downloaded, so llama.cpp
${Y}│${N} cannot race this model yet. Everything else works.
${Y}│${N}
${Y}│${N} Reason: ${GGUF_WHY:-unknown}
${Y}│${N}
${Y}│${N} Usually this is a network problem — try again in a moment.
${Y}│${N} It looks for a .gguf in ${B}$GGUF_REPO${N}
${Y}│${N} and saves it into
${Y}│${N}   $MODELS
${Y}╰───────────────────────────────────────────────────────${N}

EOF
  exit 2
fi

cat <<EOF

${G}╭─ Done ────────────────────────────────────────────────${N}
${G}│${N} ${B}$NAME is installed.${N}
${G}│${N}
${G}│${N} Open the arena and pick it from the ${B}model${N} dropdown at the top —
${G}│${N} no restart needed, it swaps over in about a minute.
${G}│${N}
${G}│${N} ${D}Not running yet?  ~/hive-limited/start.sh${N}
${G}╰───────────────────────────────────────────────────────${N}

EOF

printf '  %sInstalled models in %s:%s\n' "$D" "$MODELS" "$N"
ls -1 "$MODELS" 2>/dev/null | sed 's/^/     /'
echo
