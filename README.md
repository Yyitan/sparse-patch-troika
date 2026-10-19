English | [简体中文](README_zh.md)

# Sparse Patch on Troika: A Study of Patch-Level Local Evidence for Compositional Zero-Shot Learning (CZSL)

This repository is the public release of a research workspace on Compositional Zero-Shot Learning (CZSL) built on [Troika](https://github.com/bighuang624/Troika) (CVPR 2024, arXiv:[2303.15230](https://arxiv.org/abs/2303.15230)). The research question: **do the patch tokens of CLIP ViT contain local attribute/composition evidence beyond the CLS global feature? And can such evidence be reliably converted into gains in CZSL composition ranking via sparse pooling, object-conditioned residuals, or supervised local attribute Queries?** All experiments were run on UT-Zappos (closed-world) under an experimental protocol of dual-seed (seed0/seed1) paired controls with pre-registered validation gates: a method was allowed to read the test set only after passing a validation-only gate, and the conclusions recorded in this repository come overwhelmingly from validation (the few fixed-configuration test reads are explicitly flagged in the documentation). The `code/` directory is the final consolidated iteration of the workspace (internal codename code_v8) and contains fully runnable code for the controlled experiments on patch pooling and object-mask mechanisms; the later Query variants mentioned in the documentation (OCAQ/TGAQ/MSAQ/MPAQC/ICAQ, etc.) were implemented in their own isolated workspaces and are not included in this repository, with their experiment records and evidence fully preserved in `docs/`, `results/`, and `diagnostics/`.

This study is dominated by negative results: most of the exploration directions failed the pre-registered dual-seed validation gates. We retain these results faithfully, in the hope that they serve as a reference on the availability of patch-level local evidence in CZSL.

## Research Content

All numbers below come from the genuine records in the repository's documents (see the md files containing the tables for details).

### 1. Sparse patch pooling as a replacement for CLS visual scoring (negative)

Replaced the CLS visual scoring of Troika's three branches with class-conditioned patch max, top-k log-mean-exp pooling, or multi-layer patch features (`model/patch_pooling.py`). Paired comparisons P1–P12 showed that all max / top-k variants fell below the retrained CLS baseline (e.g., top-k=2 validation AUC 49.1946 vs. CLS 50.5867); although layer-6 top-k2 fusion gave a +0.7107-point validation gain on a fixed checkpoint, end-to-end training (P8) instead lost 1.0333 points; cross-seed paired comparisons (P9/P10) averaged 1.1073 points lower in validation AUC. We also found that hard top-1 patch selection was consistent under horizontal flips in only 9.68%–31.64% of cases — the instability of the selection is the fundamental difficulty of this direction. See `results/STAGE1`–`STAGE5` and `docs/EXPERIMENT_REPORT_20260916.md` for details.

### 2. Object-conditioned attribute residual and object-mask mechanism audit (negative)

Using layer-12 object responses for localization and layer-6 attribute responses for detail, a soft object-mask residual was injected into the composition branch: the fixed-checkpoint dual-seed validation gains were consistent (+0.4651/+0.3011), but end-to-end confirmation (P11/P12) showed the validation mean dropping by 1.5643 points. Subsequent controlled audits (D8/D9) showed that object masks generated from CLIP text attention lack object-semantic specificity (wrong-object / cross-image controls were on par or even higher); an audit of the CAMS latent units (C2/C3, L0–L3) found the 32 attribute units nearly duplicated (pairwise feature cosine 0.974/0.984), with oracle selection gains of only +2.43/+1.74 points. This entire direction was closed. See `results/STAGE6`–`STAGE10` for details.

### 3. Supervised local attribute Query series (some mechanisms effective; the final fusion failed the gate)

A series of pre-registered architecture iterations: OCAQ (object-conditioned attribute Query; supervised query specialization succeeded but the attribute residual yielded only +0.09/+0.06 points of attribute accuracy) → TGAQ (attribute text as Query; local micro below Troika's original attribute head on both seeds, with object regions shown to lack semantic specificity and attribute-only outputs collapsing onto high-frequency items) → MSAQ (layer-12 semantic keys + layer-6 detail values + sparsemax attribute Query, without object masks; sparsity 37%–54% with no local macro collapse; seed0 AUC +0.7096 points but seed1 only +0.0704 points, so the dual-seed gate failed) → MPAQ/MPAQC (within-attribute object correction; P1 +0.2780 points but P2 only +0.0089 points) → DAPQ (a mismatched-image audit revealed the y-th Query carrying a label-identity shortcut) → ICAQ (image-conditioned Query; the evidence was audited as genuinely image-dependent, but fusion AUC was +0.1690/-0.1244 points) → ICAQA (unfreezing Adapters on the last two ViT blocks with online distillation; AUC still dropped 0.2336 points in one round). The key retained conclusion: **attribute-first sparse patch representations can learn real, non-collapsed local attribute evidence; but this evidence cannot be stably converted into composition ranking gains through attribute-level additive residuals.** See `results/OCAQ_STAGE`, `results/ATTRIBUTE_EXPERT_V2`, `results/DAPQ_V6_STAGE`, `results/ICAQ_V7_STAGE`, and `results/ICAQA_V8_STAGE` for details.

### 4. Error-structure, foreground, and complementarity audits (attributes are the main bottleneck)

Zero-training audits localized the problem: 75%–80% of Troika's composition errors are of the "correct object, wrong attribute" kind, and the true composition appears in the Top-10 of more than 91% of the error samples, confirming that attribute judgment (rather than object recognition or candidate recall) is the main bottleneck; the dominant confusion is multiple materials being predicted as Leather (high-frequency prior collapse). F0b/D29/D30 showed that on the white-background UT-Zappos, MSAQ attribute attention already implicitly focuses on the true foreground (foreground quality 76.07%/94.55% vs. 44.45% for uniform), so spatial localization is not the bottleneck. The core fixable intersection between the frozen ICAQ expert and Troika's key errors is only 2.0535%/1.5868%, insufficient to support relation re-ranking training (the RCR/ADPE dual-seed gates failed). See `results/FOREGROUND_AUDIT_20260917.md`, `results/RCR_OPPORTUNITY_20260918.md`, and `results/ADPE_CONFUSION_20260918.md` for details.

## Repository Structure

```
sparse-patch-troika/
├── README.md                     # This file
├── DECISIONS.md                  # Full decision log (pre-registered gates, close/go decisions)
├── PROGRESS_20260916.md          # Chronological experiment progress records
├── code/                         # Final main code (workspace code_v8 iteration, based on Troika)
│   ├── model/                    #   Troika model + patch_pooling / object_conditioned
│   ├── clip_modules/             #   CLIP ViT-L/14 implementation (incl. BPE vocab)
│   ├── config/troika/            #   UT-Zappos / MIT-States / C-GQA training configs
│   ├── train.py / test.py        #   Training and closed/open-set evaluation entry points
│   ├── experiment.py             #   Controlled-experiment entry point with fixed train/eval modes and text chunking
│   ├── *_sweep.py                #   Fixed-checkpoint layer/fusion/mask controlled diagnostic scripts
│   ├── tests/                    #   Unit tests for pooling / object-conditioned / branch scoring
│   └── LICENSE                   #   MIT (inherited from Troika/DFSP)
├── docs/                         # Pre-registered experiment protocols per stage and the final report
├── results/                      # Per-stage result records (STAGE*.md etc.)
│   └── evidence/                 # Raw diagnostic evidence JSON (config/provenance/result)
└── diagnostics/                  # Stage 1–10 fixed-checkpoint diagnostic output JSON
```

## Reproduction

The `code/` in this repository is a controlled modification of the official Troika code (keeping the Adapter and CMT, replacing/instrumenting the visual scoring path). Reproduction steps:

```bash
# 1. Clone the official Troika repository
git clone https://github.com/bighuang624/Troika.git
cd Troika

# 2. Overwrite the official code/ directory with this repository's code/
cp -r /path/to/sparse-patch-troika/code/* code/

# 3. Environment and data preparation (CLIP ViT-L-14 weights must be downloaded manually)
conda create --name troika python=3.7
conda activate troika
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r code/requirements.txt
# See code/README.md for CLIP weights and dataset downloads

# 4. Training (the original Troika CLS baseline, i.e. the P0/P9 configs in the docs)
cd code
CUDA_VISIBLE_DEVICES=0 python -u train.py \
    --clip_arch <CLIP_MODEL_ROOT>/ViT-L-14.pt \
    --dataset_path <DATASET_ROOT>/ut-zappos \
    --save_path <SAVE_ROOT>/ut-zappos \
    --yml_path ./config/troika/ut-zappos.yml \
    --num_workers 10 \
    --seed 0

# 5. Evaluation (closed-set; for open-set see code/README.md)
python -u test.py \
    --clip_arch <CLIP_MODEL_ROOT>/ViT-L-14.pt \
    --dataset_path <DATASET_ROOT>/ut-zappos \
    --save_path <SAVE_ROOT>/ut-zappos \
    --yml_path ./config/troika/ut-zappos.yml \
    --num_workers 10 \
    --seed 0 \
    --load_model <SAVE_ROOT>/ut-zappos/val_best.pt
```

Notes:

- P0/P9 (the CLS control) in the documentation refers to the training results of the official configuration above; the patch pooling variants are switched via the configuration in `model/patch_pooling.py` (max / top-k / whether CLS participates).
- The object-mask mechanism control diagnostics (Stage 8, D8/D9) use `object_conditioned_control_sweep.py`; the layer/fusion sweeps use `layer_sweep.py` / `fusion_sweep.py`.
- Unit tests: `python -m pytest tests/` (or run files individually).
- Caution: the original `train.py` does not restore `model.train()` after validation (see the 2026-09-16 entry in `DECISIONS.md`); `experiment.py` fixes this issue and was used for all controlled comparisons.

## Citation

The code in this repository is based on Troika. If you use it, please cite the original Troika paper:

```
@inproceedings{Huang2024Troika,
    title={Troika: Multi-Path Cross-Modal Traction for Compositional Zero-Shot Learning},
    author={Siteng Huang and Biao Gong and Yutong Feng and Min Zhang and Yiliang Lv and Donglin Wang},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2024}
}
```

## License

The original Troika code is released under the MIT license (its implementation in turn references [DFSP](https://github.com/Forest-art/DFSP), also MIT-licensed); the LICENSE file in this repository's `code/` is inherited from the official Troika repository. The newly added code and modifications in this project are likewise released under the MIT license. This repository contains only the new modules and controlled modifications layered on top of the official Troika code, so that upstream licensing is respected, and all experimental numbers in the research documents and result records come from the genuine records in the repository's documents and evidence files.
