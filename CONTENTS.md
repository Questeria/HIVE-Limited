# HIVE Limited — the allow-list

**Nothing enters this repository unless it is named below.** Curation is by allow-list,
never by deleting from a copy of the full tree: a deleted file is one `git log` away, and
withheld material must never appear in *any* commit here.

Status: **packaging record.** Owner reviewed the withholding decisions and gave the
publish go-ahead 2026-08-02.

---

## SHIPS

| path | what | why it is safe |
|---|---|---|
| `LICENSE` | HIVE Limited NC v1.0 | — |
| `README.md` | limitation notice + usage | — |
| `QUICKSTART.md` | beginner path — assumes no terminal, no GitHub, nothing installed | — |
| `install.sh` | one-command installer: checks, fetches, model, launcher | touches only `~/hive-limited` and `~/.hive-limited`; every failure names the fix |
| `install.ps1` | the same installer for native Windows (no WSL needed) | same folder discipline; ASCII-only because it is piped through `irm \| iex` |
| `doctor.sh` | one-command diagnostic: every check prints PASS or the exact fix | reads local state only; its output doubles as a complete bug report |
| `CONTENTS.md` | this file | transparency is itself a feature |
| `engine/limited.py` | **purpose-written** minimal B=1 decode driver | see below — NOT the real driver |
| `engine/cudrv.py` | ctypes libcuda binding | mechanical FFI, no IP |
| `kernels/*.ptx` | certified per-32 decode lane, **prebuilt** | PTX only; see the no-source rule |
| `kernels/MANIFEST.json` | sha256 of each PTX + entry names | lets a user verify what they run |
| `serve.py` | local single-user demo surface (loopback, one lock, no queue) | a viewer, not a serving layer — see below |
| `ui/index.html` | the arena UI, **copied** from the internal build | presentation only; demo/replay mode and every baked figure removed, and the removal is enforced — see below |
| `ui/setup.html` | "add an engine" page — install, connect, bring your own model | instructions only; names no kernel, lane or spec |
| `engine/rivals.py` | adapters for rival engines the USER installs | speaks the OpenAI streaming API; contains no HIVE algorithm |
| `setup/*.sh` | installers for llama.cpp / vLLM, and a model downloader | they clone and build third-party projects on the user's machine; nothing third-party is redistributed here |
| `docs/ENGINES.md` | how to race, bring your own engine, bring your own model | — |
| `engine/reference.py` | independent numpy reference | the verification story; shares no code path with the engine |
| `engine/qwen3_tokenizer.py` | standard-library Qwen3 BPE | implements a PUBLISHED spec (tokenizer.json); leaks no HIVE algorithm |
| `engine/tokenizer_min.py` | one-instance tokenizer accessor | trivial |
| `bench.sh` + `bench.py` | measures and prints, nothing baked in | — |
| `verify.sh` + `verify.py` | reproducibility check vs the reference | the differentiating axis |
| `docs/VERIFY.md` | how the identity check works | explains the moat without giving the engine |
| `docs/UI_PORT.md` | what the UI port kept, removed, and why | records the boundary; names no internal file |
| `docs/USING_THE_ENGINE.md` + `examples/ask.py` | the engine as a library: API + runnable example | same B=1 surface the demo uses; adds no capability |
| `docs/img/*.png` | screenshots of the demo (named files only) | pictures of the shipped UI, nothing more |
| `.gitignore` + `.gitattributes` | the allow-list itself + LF normalisation | the allow-list is per-file: nothing ships unnamed |
| `REFERENCE.md` | dated **full-engine** results, wins and losses | results only — no kernel, lane, flag, spec or method that would let them be rebuilt |

## DOES NOT SHIP — and why

| withheld | reason |
|---|---|
| the measurement ledger | **the crown jewels.** Which levers work, which don't, and why. Worth more than the kernels: without it a competitor re-derives every dead end |
| the compiler's private GPU backend | the published compiler exposes only a small public subset of its GPU intrinsics; the private backend is what makes these kernels compilable at all |
| **all kernel SOURCE** | ⚠️ the subtle one. Source is uncompilable without the backend — but it still hands over the *algorithms* for a CUDA port. Reading is the easier theft. Only compiled PTX ships |
| the batch / serving path | the commercial capability. Demo is B=1 only |
| the faster unadjudicated quantization lane | unadjudicated arms never ship |
| all tensor-core frontier work | the frontier |
| the newer certified quantization stack | certified, but it is current margin |
| the production driver | contains every lane, every flag, every arm in one file — shipping it defeats the whole exercise |

## Why `engine/limited.py` is written fresh rather than stripped

Stripping the production driver is the wrong method twice over: what is removed stays
visible in structure (dead branches, flag names, the shape of what is missing), and one
missed branch leaks a lane. A purpose-written driver has no such surface — it can only do
what it was written to do, and a reader learns nothing about what else exists.

It implements exactly: load model → quantize to the certified per-32 int4 form → resident
decode loop at B=1 → greedy sampling. No batch, no env-gated arms.

## Why `ui/index.html` is COPIED, and how the copy is kept safe

The opposite decision from `engine/limited.py`, on purpose. The driver is written fresh
because a stripped driver leaks its own shape. A **user interface** has no such secret: it
is presentation, the owner asked for the same interface, and a reimplementation drifts from
it with every change.

So the arena UI is copied — and the risk that creates is answered mechanically. The copy is
produced by a transform script (kept internal, since it names internal paths) that applies a
fixed list of edits, and **every edit asserts the exact number of sites it matched**. An
edit that silently matches zero sites, or two where one was meant, aborts the port. The
output is then re-scanned and refused if it still contains demo mode, a replay figure, a
baked ms/token number, or the compiler's name.

That is the whole safety argument for copying: not "we looked carefully", but "it cannot be
built unless the removals happened."

**Removed from the copy:** demo/replay mode entirely (it animated pre-recorded timings this
repo cannot reproduce, and removing it removed the file's only two baked figures); the
compiler from the stack diagram (this build ships prebuilt PTX and no compiler).

**Added to the copy:** the limitation banner above everything, per LICENSE §6(c); a solo-run
path, because a fresh clone has no rival installed and the old code returned early when
either lane was empty — the first LAUNCH a new user pressed would have shown nothing at all;
and a `/engines` query so our own bytes-per-token comes from the loaded weights instead of a
constant, and so a missing rival is labelled in the dropdown rather than failing on click.

**Changed in the copy, and why each was necessary:**

- **The claim board STAYS, carrying full-engine results** (owner decision, 2026-08-01; I had
  first removed it). Two engines now appear on one page — a live race of *this* build above,
  a table of the *full* engine below — so the boundary is made impossible to miss rather than
  left to inference: an amber banner directly above the table states that everything below it
  is the full engine, not the build just raced, and that the figures are not reproducible from
  this repository. The rows are the same set as `REFERENCE.md` and held to the same rules —
  claim-grade only, losses beside wins (NVIDIA leads four rows), bit-width asymmetry stated,
  and any row whose correctness adjudication is still open says so in its own basis line.
- **The hardware panel reads YOUR machine.** It was a fixed spec sheet for the laptop the
  recorded numbers came from — a different computer than the reader's, describing runs this
  build does not contain. It is now read live at request time, the GPU fields through the
  same `libcuda` binding the engine uses.
- **`countdown()` no longer depends on `requestAnimationFrame` alone.** A hidden tab
  suspends rAF, so its promise never resolved: start a race, switch tabs, come back to a
  dead page with LAUNCH disabled and no recovery but a reload. `replayLane` already guarded
  against exactly this ("not rAF: survives tab-backgrounding so the race can't freeze"); the
  countdown had been missed. Now rAF for smoothness, a timer for the guarantee.
- **`lastRun` captured the wrong string.** `L.buf` is a *drain* buffer — the render tick does
  `L.text += L.buf; L.buf = ''` — so the answer accumulates in `L.text` while `L.buf` holds
  only the tokens since the last repaint. `lastRun.text = LU.buf` therefore stored a fragment
  (measured: 12 characters of a 541-character answer) or an empty string. That is the string
  the reproducibility flow re-compares a second run against, so the build's headline claim
  was checking the wrong bytes. Now `L.text + L.buf` — drained plus undrained.

## Racing rivals — why this does not become a claim

The arena runs a rival **the user installed, on the user's machine**. We ship the harness;
they produce the result. That keeps the standing rule intact — this build makes no
comparative claim — while letting anyone check ours against theirs.

Guards that keep it honest rather than flattering: cold model load is reported and excluded
from every timing; both lanes are timed by one clock in this process; both engines run
greedy; rivals launch with ordinary flags, listed in `setup/`; the bit-width difference is
stated on the page, including that this build's 5.00 bpw is **more** bytes than a 4-bit
rival; and when no rival answers, the result card says no winner is claimed instead of
declaring one against an engine that never ran.

## Why graph replay is shipped, and why it gives nothing away

HIVE Limited was losing to llama.cpp. Profiling said why, and it was not the kernels: a
decode step issues hundreds of kernel launches, each costing host time through ctypes —
the bulk of every step was spent in Python while the GPU waited. The engine was
host-bound.

Replaying that fixed sequence from a **CUDA graph** removes the host from the loop, with
tokens byte-identical. The demo measures and prints the before/after itself at every
startup — the numbers belong to your machine, not this page.

This is safe to ship, and the reason is worth stating precisely:

- **A graph is a recording of launches the driver already saw.** It contains no kernel
  source, no algorithm, no quantization spec. Reading it back tells you the same thing
  watching the process would: that these PTX entry points ran in this order.
- **Capture is public NVIDIA driver surface** (`cuStreamBeginCapture`, `cuGraphLaunch`).
  Every serious engine uses it, including the ones raced here. Knowing we do is not a
  disclosure; it is the default assumption.
- **It works here only because of a property the build already had** — argmax writes the
  next token, position and step counter on the *device*, so a replay advances state with no
  host round-trip. That design was already visible in `limited.py`.

What still is not here, and is what the speed actually rests on in the full engine: the
kernels, the compiler that emits them, the coarser-scaled weight lane, batching, and the
measurement ledger that says which levers were worth building.

**The fast path is earned, not assumed.** At load the server runs the same prompt through
both paths and compares 24 tokens; the graph is used only if they are identical, and a
mismatch disables it and says so. An engine that is fast and wrong is worse than one that
is slow.

## The no-baked-numbers rule — and its one deliberate exception

**The rule, unchanged:** no performance number is baked into the *software*. `bench.sh`
and `verify.sh` print only what they measured on the user's machine, in that run. The
engine and the UI contain no figures at all. A number a reader cannot reproduce is a
claim, not a demonstration.

**The exception, added deliberately (2026-08-01):** `REFERENCE.md` carries dated results
from the **full engine**, which is not in this repository and cannot be run from it.

The two are different kinds of statement and the repo keeps them physically apart:

| | this build's numbers | `REFERENCE.md` |
|---|---|---|
| where | printed at runtime | one static page |
| status | demonstrated | asserted, dated, on our word |
| reproducible here | yes, that is the point | **no**, and it says so |

The exception exists because the omissions are severe enough that a reader could
reasonably take this build's speed for the engine's ceiling — which would be its own kind
of dishonesty. What keeps it disciplined:

- **results only.** No kernel name, lane, flag, tile shape, spec or method appears — the
  page states what was measured, never how it was achieved.
- **losses published beside wins.** The H100 table shows where NVIDIA is ahead, including
  by ~3×. A win-only table would misrepresent the engine.
- **only claim-grade figures.** Same-round interleaved medians with gate counts,
  token-gated against an independent oracle. No projections and no "up to".
- **withdrawn results stay withdrawn.** Several earlier HIVE figures were retired when
  their measurement method proved faulty; none of them appear.
- **bit-width asymmetry disclosed** where a comparison is not equal-bit-width.

Comparative results are still not shipped as *this build's* claims. If a user wants a
comparison they run the rival themselves on the same box, which is the only comparison
worth trusting and is how every figure on that page was established.

## Pre-flight checklist — completed 2026-08-02, the night of publication

- [x] every path in the tree appears in the SHIPS table above
- [x] `git log -p` contains no withheld path in *any* commit — the public history is a
      single fresh root, authored under the project identity, created for exactly this
      guarantee (the development history stayed private)
- [x] no kernel-source file anywhere in history
- [x] no benchmark figure baked into any *code* path; `bench.sh` output is runtime-only
- [x] `REFERENCE.md` is the ONLY static-number file, states it is not reproducible here,
      names no kernel/lane/flag/spec, and publishes losses beside wins
- [x] every figure in `REFERENCE.md` is claim-grade and not on a withdrawn list
- [x] a fresh `git clone` can actually run everything it ships — verified on a bare
      clone the same night: modes intact, every script's counterpart present, doctor
      all-green, the library example generates (this failed once historically: both
      `.sh` were tracked while their `.py` were not)
- [x] README limitation banner present and first on screen
- [x] LICENSE §6 (no misrepresentation, no attributed benchmarks) intact
- [x] two independent audit passes over every shipped file and the full history
- [x] owner review and explicit go-ahead (2026-08-02)
