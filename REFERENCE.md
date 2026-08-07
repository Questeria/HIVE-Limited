# HIVE — measured results for the full engine

**Last updated: 7 August 2026.**

This page exists because HIVE Limited is deliberately slower and narrower than HIVE, and a
demo you can run should not be mistaken for the ceiling. Every figure below was measured on
the **full engine**, which is **not** included in this repository.

**These figures are not reproducible here.** The code in this repository cannot produce
them, and nothing on this page should be quoted as a HIVE Limited result. HIVE Limited is
batch-1 only, ships one quantization form, and has no optimized prefill path — see the
README for what it omits.

Everything else in this repository is a demonstration: you run it and watch it measure
itself. This page is not that. It is an assertion, dated, with the protocol and the losses
stated so you can judge how much weight to give it.

---

## What is being compared

| | |
|---|---|
| **HIVE** | own compiler → PTX, driver only. No CUDA toolkit, no cuBLAS, no cuDNN, no TensorRT, no PyTorch. |
| **Protocol** | same-round interleaved medians, both engines alternating on one machine, gates counted and reported. Single-run and non-interleaved numbers are not used. |
| **Correctness** | every result token-gated against an independent oracle. A run whose tokens diverge is discarded, not published. |

---

## RTX 3070 Laptop — the exact machine

Stated in full, in the same fields the demo's hardware panel reports, so you can compare
your machine against the one these numbers came from rather than guessing from a model name.

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3070 **Laptop** GPU — 8 GiB GDDR6, sm_86, 40 SMs |
| memory bandwidth | **384 GB/s** theoretical (6001 MHz × 256-bit); **~359 GB/s** achieved |
| CPU | Intel Core i9-10980HK — 8 cores / 16 threads |
| RAM | 31.8 GiB |
| chassis | ASUS ZenBook Pro Duo 15 OLED (UX582LR) |
| OS | Windows 11 + WSL2 |

Two notes that matter if you are checking these figures:

- **This is the laptop part, not the desktop 3070.** Its memory runs at 12 Gbps against the
  desktop card's 14, so peak bandwidth is **384 GB/s, not 448**. Decode is memory-bound, so
  quoting the desktop figure would understate how close to the ceiling these results sit.
- **Percentages below use the *achieved* bandwidth (~359 GB/s), measured on this card** —
  never the datasheet peak.

### The results

### vs vLLM — Qwen3-4B, **both engines at int4** (equal bit-width)

| Axis | Result | Date |
|---|---|---|
| Prefill @1024 | **HIVE 1.49×** (1.38× on the conservative pairing) | 2026-07-24 |
| Decode | **HIVE 1.3–1.8×** | 2026-07-21 |
| 8B capacity | **HIVE runs 8B int4 with full CUDA graphs in 8 GB; vLLM cannot** (OOMs, eager-only) | 2026-07-21 |

### vs llama.cpp — Qwen3-8B

| Axis | Result | Date |
|---|---|---|
| Prefill @1024 | **HIVE 1.34×** (20 rounds, 60/60 gates) | 2026-07-20 |
| Decode | **HIVE 1.25–1.26×** (12 rounds, 36/36 gates) | 2026-07-20 |
| Prefill @2048 | **HIVE 1.16–2.50×** (the spread is the rival's, under sustained load) | 2026-07-20 |

> **Bit-width disclosure:** this pairing is **not** equal bit-width. HIVE's form here is
> ~4.2 bits/weight against llama.cpp's ~5.0–5.3, so part of the margin is the format, not
> the engine. The vLLM comparison above **is** equal-bit-width and is the fairer read.

### vs the hardware itself

| Axis | Result | Date |
|---|---|---|
| Decode | **~80% of the measured memory roofline** | 2026-07-20 |
| Prefill | ~38% of the measured compute ceiling | 2026-07-20 |

Measured on the card, not taken from a datasheet — the sustained figures are materially
below the published peaks, and the published peaks are not used as denominators anywhere.

---

## H100 SXM5 — the exact machine

| | |
|---|---|
| GPU | NVIDIA H100 **SXM5** — 80 GB HBM3, sm_90, 132 SMs |
| host | rented cloud pod, one GPU, network-attached volume |
| rival | TensorRT-LLM 1.2.1, W4A16-AWQ g128, CUDA graphs on |

Not the PCIe H100 and not a MIG slice — the SXM5 part, which has the higher bandwidth and
power budget of the two. If you reproduce this on a PCIe card, expect different numbers.

### Qwen3-1.7B vs TensorRT-LLM, 3 interleaved rounds

| Batch | HIVE | TensorRT-LLM | Result | Label | Date |
|---|---|---|---|---|---|
| B = 1 | 594.8 tok/s | 494.6 tok/s | **HIVE 1.20×** | ✅ clean | 2026-07-30 |
| B = 4 | 1765.2 tok/s | 1562.6 tok/s | **HIVE 1.13×** | ⚠ open | 2026-07-30 |
| B = 8 | 2891.7 tok/s | 2705.7 tok/s | **HIVE 1.07×** | ⚠ open | 2026-07-30 |
| B = 16 | 3764.9 tok/s | 4555.1 tok/s | **TensorRT-LLM 1.21×** | — | 2026-07-30 |
| B = 32 | 4227.8 tok/s | 8312.5 tok/s | **TensorRT-LLM 1.97×** | — | 2026-07-30 |
| B = 64 | ≈4.7k tok/s | ≈14.7k tok/s | **TensorRT-LLM 3.1×** | — | 2026-08-02 |
| TTFT @512 | 17.5 ms | 6.23 ms | **TensorRT-LLM 2.8×** | — | 2026-07-30 |

**On the labels.** B=1 is the **certified** lane: its correctness adjudication is complete and
clean, so the headline batch-1 win stands on its own. **B=4 and B=8 come from a faster lane
whose adjudication is still open** — one outstanding case — so they are reported with that
stated rather than presented as settled. A previous version of this page omitted that
distinction; it was wrong to.

**We publish the losses.** HIVE leads at small batch on NVIDIA's own datacenter silicon and
falls behind as batch grows; prefill on H100 is currently NVIDIA's by a wide margin. Both are
open engineering fronts, and a table showing only the wins would misrepresent where the engine
stands.

**The moving edge — research, not yet record.** Measured 2026-08-02 on the same pod, with
token-identical output at every point: a tensor-core decode lane raised HIVE's B=64 to
≈7.3k tok/s (gap ≈2.0×), and a tensor-core prefill lane brought TTFT@512 to 14.7 ms
(gap ≈2.4×). Both are pre-certification research lanes; they move into the table above only
when their certification completes — and if it fails, they will not.

---

## Portability

Token-identical output across **three NVIDIA generations** — sm_86, sm_89, sm_90 — from the
same compiler-emitted PTX, with no per-architecture source. Verified by full identity
batteries on each.

---

## Model families — added 6 August 2026

Portability across GPUs is one axis; portability across model *architectures* is another.
HIVE has now been tested on **six models across five architecture families**: Qwen3, Llama 3.2, GPT-2, Phi-3.5 and
Gemma 3. HIVE Limited ships Qwen3 only.

**Correctness — the binding gate.** Every model must generate output **token-identical to
an independent reference implementation** of the same quantization spec. That is the same
kind of check `./verify.sh` runs here on Qwen3, applied to each family. Most recent full
pass: **48 of 48 test legs across six models, zero failures** (7 August 2026).

The sixth model is the one worth mentioning: it was picked *after* the engine was built,
to check that a model nobody had designed around would run. It did, unmodified.

Where the model is genuinely undecided between its top two candidates — within a factor of
two in probability — the run is recorded as a **disclosed near-tie**, not counted as
agreement. Papering over those would inflate the result.

**Speed against llama.cpp, same machine, same protocol as everything else on this page:**

| model | HIVE ÷ llama.cpp |
|---|---|
| Llama-3.2-3B | **1.46×** |
| Qwen3-8B | ~1.1–1.25× |
| Llama-3.2-1B | **1.07×** |
| Phi-3.5-mini | 0.94× |
| GPT-2 | 0.86× |
| Gemma-3-4B | 0.60× |

**Three of those six are losses, and they are published because they are true.** Wins and
losses were measured identically: same-round interleaved medians, failed runs counted
rather than discarded. Bit widths differ and are disclosed — HIVE decodes ~4.4 bits per
weight against Q4_K_M's ~4.7, so the raw ratios do not flatter HIVE.

Size does not explain the split: the 1B model wins and the 4B model loses hardest. The
losses track a narrower engineering gap, and until it is closed they stay on this page.

GPT-2 is the smallest model here and the one where per-token overhead matters most; it is
a loss and is listed as one.

---

## Preview the unrestricted HIVE

The full engine is not public. If you want to see it run, discuss licensing, or ask anything
about the numbers on this page:

📧 **ajdemarco10@gmail.com**

---

### Standing rules for this page

- Figures are dated, and the date is the measurement date, not the publication date.
- No projection, extrapolation, or "up to" figure appears here — only measured medians.
- Withdrawn results are removed rather than footnoted. Several earlier HIVE figures were
  retired after their measurement method was found faulty; none of them appear above.
- Losses are listed beside wins.
