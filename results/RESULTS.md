# Results

## E0: copied-code baseline validation

- Date: 2026-09-15
- Dataset: UT-Zappos
- Seed: 0
- Setting: closed world
- Checkpoint: historical Troika reproduction `val_best.pt`
- GPU: 1

| Seen | Unseen | HM | AUC | Attr | Obj |
|---:|---:|---:|---:|---:|---:|
| 65.0049 | 73.9820 | 55.3891 | 41.5095 | 50.3775 | 72.4777 |

The modified default CLS path exactly matches the newly evaluated original
code. Relative to the historical summary, AUC differs by +0.010700 percentage
points (historical 41.498754), and attribute accuracy by -0.068635 points.
S/U/HM/Obj agree. The historical checkpoint is loadable; these small differences
are recorded, not asserted to be zero. Historical training also contains a
train/eval mode bug, so new variants require a freshly trained matched control.

## Historical pending list (completed)

- E1a: patch max including CLS.
- E1b: patch max excluding CLS.
- E2: top-k log-mean-exp pooling.

## 2026-09-16 verification

- Four CPU pooling tests passed. The original gradient test used an exactly
  aligned cosine vector, whose derivative is zero; a nondegenerate input fixes
  that test without changing the pooling formula.
- Real-image smoke test passed for CLS, max+CLS, max without CLS, and top-k=2.
- Each mode had finite loss and finite, nonzero visual-adapter gradients.
- Direct and cached-text logits agreed exactly in the smoke test (max error 0).
- Tiny-batch peak CUDA allocation: 3,266,823,168 bytes. This is not full-batch memory.
- This early smoke-test note is superseded by the controlled training reports.

## Current result index

All P0-P12 training and diagnostics are complete. The final object-conditioned
attribute residual result is negative on both paired validation seeds: -1.5752
points for seed0 and -1.5534 for seed1. See `STAGE7_20260916.md` and
`../docs/EXPERIMENT_REPORT_20260916.md` for the complete analysis.
