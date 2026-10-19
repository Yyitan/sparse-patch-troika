# Experiment Plan

## Primary metrics

Report Seen, Unseen, HM, AUC, attribute accuracy, and object accuracy for
closed and open world. Patch changes are judged before adding memory or other
test-time modules.

## Stage 0: trusted baseline

- E0 evaluates the existing Troika seed-0 checkpoint from the new workspace.
- Expected UT-Zappos closed result: S 65.00, U 73.98, HM 55.39, AUC 41.50.
- A mismatch larger than numerical noise blocks later experiments.

## Stage 1: hard sparse alignment

- E1a: class-conditioned max over CLS and patch tokens.
- E1b: class-conditioned max over patch tokens only.
- Train seed 0 first. Expand to three seeds only after a useful validation
  AUC/HM result.

## Stage 2: multi-token sparse pooling

- Use normalized top-k log-mean-exp pooling.
- Test k in 2, 4, and 8 with temperature 0.07.
- k=1 must be numerically equivalent to hard max.

## Stage 3: visual hierarchy

- Compare final-layer, middle-layer, and fused middle/final patch tokens.
- Track primitive accuracy to distinguish better attribute extraction from
  genuine composition binding.

## Stage 4: binding

- Locate objects with high-level sparse patches.
- Extract attributes from middle-level patches conditioned on object regions.
- Add spatial binding and augmentation-consistency losses.

## Resource policy

- Prefer an idle GPU.
- Use at most three GPUs concurrently.
- Start with one GPU per experiment and avoid GPU 0 when it is occupied.
- Do not write into the original Troika or CAMS directories.
