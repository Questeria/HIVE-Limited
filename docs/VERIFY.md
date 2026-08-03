# What `verify.sh` proves — and what it does not

Run it:

```bash
./verify.sh /path/to/qwen3-model
```

It makes two checks. Both matter, and neither claims more than it establishes.

## 1. Determinism

The same prompt, run twice, must produce the same tokens.

This sounds trivial and is not. Engines that select kernel tilings per shape can return
different results at different batch sizes or configurations — so "same input, same
output" is a property you have to *build*, not one you get. HIVE Limited fixes its
tiling and reduction order, so this check is expected to pass exactly, every time. If it
ever fails, something is genuinely wrong.

## 2. Agreement with an independent reference

`engine/reference.py` is a numpy forward pass on the CPU. It shares no kernel, no PTX and
no GPU code path with the engine — it quantizes the weights itself from the published
rule and runs the model in plain Python. The two must select the same tokens.

### Near-tie forks are licensed, and the threshold is principled

When the top two candidates are separated by less than **ln 2 ≈ 0.693** in logit space,
the ranking is decided *below the level either implementation can claim*. The engine
reduces across 32 lanes in a warp butterfly; numpy reduces in its own order; float
addition is not associative, so the last bits differ. At a margin of, say, 0.002, which
candidate comes first is a property of the summation order — not a fact about the model.

So a divergence **below** that threshold is reported as a licensed fork, with the margin
printed and the agreed prefix stated. A divergence **at or above** it fails, because at a
decisive margin the two implementations genuinely disagree, and that is an error.

The tool prints the margin either way, so you never have to take the verdict on trust.

## What is deliberately NOT claimed

**Not f32 bit-identity.** The logits are not asserted bit-identical between engine and
reference, and they are not. Token agreement is the claim.

**Not conformance to any other implementation.** This checks *this engine* against *this
reference*. In particular, the shipped `rmsnorm_bp` kernel uses an RMSNorm epsilon of
**1e-5**, and the reference matches it so that the comparison is meaningful. Some
published configurations for this model family specify 1e-6. That difference is small
(order 1e-6 relative at typical activation scale) but it is real, and it means these
tokens are not asserted to match a HuggingFace or PyTorch run of "the same" model.

We would rather state that plainly than let "bit-exact" quietly imply a conformance claim
we have not tested. A verification tool that overstates its scope is worse than none.

## Why this is the interesting axis

Speed claims are easy to make and hard to trust. Reproducibility is the reverse: it is
hard to build and easy to check — which is why this repository ships the checker rather
than a table of results, and why every number `bench.sh` prints comes from the run that
printed it.
