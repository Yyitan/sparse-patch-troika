# Sparse Patch on Troika：面向组合零样本学习（CZSL）的 Patch 级局部证据研究

本仓库是一项基于 [Troika](https://github.com/bighuang624/Troika)（CVPR 2024，arXiv:[2303.15230](https://arxiv.org/abs/2303.15230)）的组合零样本学习（Compositional Zero-Shot Learning, CZSL）研究工作区的公开发布版本。研究问题：**CLIP ViT 的 patch token 中是否存在 CLS 全局特征之外的局部属性/组合证据？这些证据能否通过稀疏池化、对象条件化残差或监督式局部属性 Query，稳定地转化为 CZSL 组合排序的提升？** 全部实验在 UT-Zappos（closed-world）上进行，采用双 seed（seed0/seed1）配对对照与预注册验证门槛的实验协议：方法只有在通过验证集（validation-only）门槛后才允许读取 test，本仓库记录的所有结论绝大多数来自 validation（少数固定配置的 test 读取均已在文档中显式标注）。`code/` 目录是工作区最后一次 consolidated 迭代（内部代号 code_v8），包含 patch pooling 与对象 mask 机制控制实验的完整可运行代码；文档中涉及的后续 Query 变体（OCAQ/TGAQ/MSAQ/MPAQC/ICAQ 等）在各自的隔离工作区中实现，未包含在本仓库中，其实验记录与证据完整保留在 `docs/`、`results/` 与 `diagnostics/` 中。

本研究以阴性结果为主：大多数探索方向未通过预注册的双 seed 验证门槛。我们如实保留这些结果，希望它们能为 CZSL 中 patch 级局部证据的可用性提供参考。

## 研究内容

以下数字均来自仓库内文档的真实记录（详见表格所在的 md 文件）。

### 1. Sparse patch pooling 替代 CLS 视觉评分（阴性）

将 Troika 三个分支的 CLS 视觉评分替换为 class-conditioned patch max、top-k log-mean-exp 池化或多层 patch 特征（`model/patch_pooling.py`）。P1–P12 的配对对照显示：max / top-k 变体全部低于重新训练的 CLS 基线（如 top-k=2 验证 AUC 49.1946 vs CLS 50.5867）；固定 checkpoint 上 layer-6 top-k2 融合虽有 +0.7107 点验证增益，端到端训练（P8）后反而下降 1.0333 点；跨 seed 配对（P9/P10）验证 AUC 平均低 1.1073 点。另发现 hard top-1 patch 选择在水平翻转下一致率仅 9.68%–31.64%，选择不稳定是该路线的根本困难。详见 `results/STAGE1`–`STAGE5` 与 `docs/EXPERIMENT_REPORT_20260916.md`。

### 2. 对象条件化属性残差与对象 mask 机制审计（阴性）

用第 12 层对象响应定位、第 6 层属性响应提取细节，向组合分支注入 soft 对象 mask 残差：固定 checkpoint 双 seed 验证增益一致（+0.4651/+0.3011），但端到端确认（P11/P12）验证均值下降 1.5643 点。随后的对照审计（D8/D9）表明 CLIP 文本 attention 生成的对象 mask 缺乏对象语义特异性（wrong-object / cross-image 对照持平甚至更高）；对 CAMS latent 单元的审计（C2/C3、L0–L3）发现 32 个 attribute 单元近乎重复（特征两两余弦 0.974/0.984），oracle 选择增益仅 +2.43/+1.74 点。该路线整体关闭。详见 `results/STAGE6`–`STAGE10`。

### 3. 监督式局部属性 Query 系列（部分机制有效，最终融合未过门槛）

一系列预注册架构迭代：OCAQ（对象条件属性 Query，监督 query 特化成功但属性残差仅 +0.09/+0.06 点属性准确率）→ TGAQ（属性文本作 Query，双 seed 局部 micro 均低于 Troika 原属性头，且对象区域被证明无语义特异性、属性-only 输出高频塌缩）→ MSAQ（无对象 mask 的 layer12 语义 key + layer6 细节 value + sparsemax 属性 Query，稀疏率 37%–54%、局部 macro 无塌缩，seed0 AUC +0.7096 点但 seed1 仅 +0.0704 点，双 seed 门槛未过）→ MPAQ/MPAQC（属性内对象校正，P1 +0.2780 点但 P2 仅 +0.0089 点）→ DAPQ（被错配图像审计发现第 y 个 Query 携带标签身份的捷径）→ ICAQ（图像条件 Query，证据经审计真实依赖图像，但融合 AUC +0.1690/-0.1244 点）→ ICAQA（解冻 ViT 最后两个 block 的 Adapter 并在线蒸馏，一轮 AUC 仍下降 0.2336 点）。核心保留结论：**属性-first 的稀疏 patch 表征可以学到真实、不塌缩的局部属性证据；但这些证据无法通过属性级加性 residual 稳定转化为组合排序增益。** 详见 `results/OCAQ_STAGE`、`results/ATTRIBUTE_EXPERT_V2`、`results/DAPQ_V6_STAGE`、`results/ICAQ_V7_STAGE`、`results/ICAQA_V8_STAGE`。

### 4. 错误结构、前景与互补性审计（属性是主要瓶颈）

零训练审计将问题定位：Troika 组合错误中 75%–80% 属于"对象正确、属性错误"，错误样本的真实组合 Top-10 覆盖超过 91%，确认属性判断（而非对象识别或候选召回）是主要瓶颈；主要混淆集中于多种材质被预测为 Leather（高频先验塌缩）。F0b/D29/D30 表明在白背景的 UT-Zappos 上，MSAQ 属性 attention 已隐式聚焦真实前景（前景质量 76.07%/94.55% vs 均匀 44.45%），空间定位不是瓶颈。固定 ICAQ 专家与 Troika 关键错误的核心可修正交集仅 2.0535%/1.5868%，不足以支撑关系重排训练（RCR/ADPE 双 seed 门槛未过）。详见 `results/FOREGROUND_AUDIT_20260917.md`、`results/RCR_OPPORTUNITY_20260918.md`、`results/ADPE_CONFUSION_20260918.md`。

## 仓库结构

```
sparse-patch-troika/
├── README.md                     # 本文件
├── DECISIONS.md                  # 全程决策日志（预注册门槛、关闭/放行决定）
├── PROGRESS_20260916.md          # 按时间的实验进展记录
├── code/                         # 最终主代码（工作区 code_v8 迭代，基于 Troika）
│   ├── model/                    #   Troika 模型 + patch_pooling / object_conditioned
│   ├── clip_modules/             #   CLIP ViT-L/14 实现（含 BPE 词表）
│   ├── config/troika/            #   UT-Zappos / MIT-States / C-GQA 训练配置
│   ├── train.py / test.py        #   训练与闭集/开集评测入口
│   ├── experiment.py             #   修正 train/eval 模式与文本分块后的对照实验入口
│   ├── *_sweep.py                #   固定 checkpoint 的 layer/fusion/mask 控制诊断脚本
│   ├── tests/                    #   pooling / object-conditioned / branch scoring 单元测试
│   └── LICENSE                   #   MIT（继承自 Troika/DFSP）
├── docs/                         # 各阶段预注册实验方案与总报告
├── results/                      # 各阶段结果记录（STAGE*.md 等）
│   └── evidence/                 # 诊断原始证据 JSON（config/provenance/result）
└── diagnostics/                  # Stage 1–10 固定 checkpoint 诊断输出 JSON
```

## 复现

本仓库 `code/` 是对 Troika 官方代码的受控修改（保留 Adapter 与 CMT，替换/诊断视觉评分路径）。复现步骤：

```bash
# 1. 克隆 Troika 官方仓库
git clone https://github.com/bighuang624/Troika.git
cd Troika

# 2. 用本仓库的 code/ 覆盖官方 code/ 目录
cp -r /path/to/sparse-patch-troika/code/* code/

# 3. 环境与数据准备（CLIP ViT-L-14 权重需手动下载）
conda create --name troika python=3.7
conda activate troika
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r code/requirements.txt
# CLIP 权重与数据集下载见 code/README.md

# 4. 训练（Troika 原始 CLS 基线，即文档中的 P0/P9 配置）
cd code
CUDA_VISIBLE_DEVICES=0 python -u train.py \
    --clip_arch <CLIP_MODEL_ROOT>/ViT-L-14.pt \
    --dataset_path <DATASET_ROOT>/ut-zappos \
    --save_path <SAVE_ROOT>/ut-zappos \
    --yml_path ./config/troika/ut-zappos.yml \
    --num_workers 10 \
    --seed 0

# 5. 评测（闭集；开集另见 code/README.md）
python -u test.py \
    --clip_arch <CLIP_MODEL_ROOT>/ViT-L-14.pt \
    --dataset_path <DATASET_ROOT>/ut-zappos \
    --save_path <SAVE_ROOT>/ut-zappos \
    --yml_path ./config/troika/ut-zappos.yml \
    --num_workers 10 \
    --seed 0 \
    --load_model <SAVE_ROOT>/ut-zappos/val_best.pt
```

说明：

- 文档中的 P0/P9（CLS 对照）即上述官方配置的训练结果；patch pooling 变体通过 `model/patch_pooling.py` 中的配置切换（max / top-k / CLS 参与与否）。
- 对象 mask 机制控制诊断（Stage 8, D8/D9）使用 `object_conditioned_control_sweep.py`；layer/fusion 扫描使用 `layer_sweep.py` / `fusion_sweep.py`。
- 单元测试：`python -m pytest tests/`（或逐文件运行）。
- 注意：原版 `train.py` 在验证后不会恢复 `model.train()`（详见 `DECISIONS.md` 2026-09-16 条目），`experiment.py` 已修正该问题并用于全部受控对照实验。

## 引用

本仓库代码基于 Troika。如使用，请引用 Troika 原文：

```
@inproceedings{Huang2024Troika,
    title={Troika: Multi-Path Cross-Modal Traction for Compositional Zero-Shot Learning},
    author={Siteng Huang and Biao Gong and Yutong Feng and Min Zhang and Yiliang Lv and Donglin Wang},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2024}
}
```

## License

Troika 原始代码以 MIT 协议发布（其实现又参考了 MIT 协议的 [DFSP](https://github.com/Forest-art/DFSP)），本仓库 `code/` 中的 LICENSE 文件即继承自 Troika 官方仓库。本项目新增的代码与修改同样以 MIT 协议发布；研究文档与结果记录中的所有实验数字均来自仓库内文档与证据文件的真实记录。
