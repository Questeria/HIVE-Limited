# Porting the rocket UI into HIVE Limited

The demo ships the same rocket-race surface as our internal arena, with the owner
decisions of 2026-08-01 applied. This page records what was kept, what was removed,
and why the boundary is where it is.

## 1. The rival lane stays — and races engines the user installs

The race framing is what makes the UI worth looking at, so both lanes remain. HIVE
Limited bundles **no rival engine and no comparative number**. The opposing lane runs an
engine the user installs on their own machine (`setup/` has installers for llama.cpp and
vLLM; anything with an OpenAI-compatible endpoint can be connected on a documented port).

This follows the repo's standing rule — *if a user wants a comparison they run the rival
themselves on the same box, which is the only comparison worth trusting* — and it has a
useful property: **every live comparison the UI can display is one the user produced.**
We ship the harness; we ship no claim.

## 2. Replay/mock mode is REMOVED, not relabelled

The internal UI has a replay mode that animates a previously measured per-token latency
when no GPU is present, plus copy explaining two competing margin figures. Those are
baked numbers and they do not ship.

(The figures themselves are deliberately not restated here either. A private measurement
quoted inside the public repository is still a private measurement published — and it
would be quotable as "HIVE's number" from our own demo, which is precisely what the rule
exists to prevent. This paragraph is the fix for a leak my own pre-flight check caught.)

**With no GPU, this build shows no numbers.** The page says why and offers nothing else.
Keeping the animation with an "illustrative" label was considered and rejected: a figure
on screen is a figure people quote, and this repository cannot reproduce it.

Everything the live race displays comes from the run happening in front of you.

## 3. One static exception, fenced

The page also carries the full-engine results board — the same dated rows as
`REFERENCE.md`, behind an amber banner stating that everything below it is the full
engine, not the build being raced, and is not reproducible from this repository. That is
an explicit owner decision (2026-08-01, recorded in `CONTENTS.md`), and it is the only
static content on the page: the live lanes above it never display a number they did not
just measure.

## Consequences for the port

| element | disposition |
|---|---|
| rocket race visuals, layout, styling | ported |
| live token stream | ported (SSE `/run`) |
| stack receipt panel | ported — it is the differentiating axis and it is a check, not a claim |
| measured tok/s, bits-per-weight | ported, runtime-only |
| rival backends | ported as **adapters for engines the user installs** — nothing third-party is bundled |
| replay / mock mode | **removed** |
| baked margins and `ms/token` copy | **removed** |
| scoreboard | session-local: it holds the runs you just made. Prior-result history does not ship; the full-engine board (above) is the fenced exception |

## Why the server is written fresh

The internal arena server is a large multi-rival race harness with its own claim and
provenance machinery. `serve.py` here is purpose-written for this demo: it binds to
loopback, serialises work behind a single lock, and exposes only the documented local
endpoints the page in front of you calls. Same reasoning as the engine: a purpose-written
surface can only do what it says, and a reader learns nothing about what else exists.

It has no queue, no batching, no auth. It is a viewer, not a serving layer.
