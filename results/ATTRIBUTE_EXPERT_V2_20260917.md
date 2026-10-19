# 属性专家 Query V2 实验记录

日期：2026-09-17

## D15/D16：固定 OCAQ checkpoint 属性专家诊断

目的：区分“通用 Query 本身判别不足”“Query 有互补但路由失败”和“对象预测/后验融合损失”三种原因。只使用 validation，不读取 test。

| 指标 | seed0 | seed1 |
|---|---:|---:|
| Troika 原属性头 | 34.8164% | 33.2607% |
| 真实对象聚合局部属性 | 28.9359% | 35.4076% |
| 预测对象聚合局部属性 | 29.9004% | 35.6565% |
| 对象后验聚合局部属性 | 29.9627% | 35.5943% |
| 路由最大 Query | 29.0292% | 35.4698% |
| 任一 Query 正确 oracle | 31.0516% | 36.6210% |
| oracle 相对聚合提升 | +2.1157 | +1.2134 |
| 真实对象局部相对原属性头 | -5.8805 | +2.1469 |

四个单 Query 的准确率在 seed0 为 28.56%-29.31%，在 seed1 为 35.28%-35.63%，彼此非常接近。任一 Query oracle 的上界只增加 2.12/1.21 个百分点，两个种子都低于预注册的 3 点门槛，因此没有足够证据支持继续训练更复杂的 Query router。

对象条件不是当前主瓶颈：使用预测对象或对象后验并没有比真实对象下降，反而略高。这说明继续扩大 top-M 对象候选不能解决属性问题。

类别级结果显示明显长尾偏置。Faux Leather、Full grain leather、Nubuck、Patent Leather 等属性在大部分设置中接近 0；模型主要依赖 Leather 等高频属性。seed0 局部 Query 对 Synthetic 有一定改善，但明显损失 Leather、Satin 和 Suede；seed1 对 Leather/Satin 改善，但 Synthetic 和 Nylon 大幅下降，跨种子不稳定。

### D15/D16 决策

按预注册规则，下一版不做属性条件 router，而改成 **Text-Grounded Attribute Query（TGAQ）**：每个属性文本自身作为 Query，在对象区域及其邻域内寻找对应视觉证据。第一阶段只训练属性专家，并使用固定的类别平衡交叉熵；不训练组合 gate，不读取 test。只有双种子局部属性准确率都超过各自 Troika 原属性头至少 1 个百分点，才允许进入 CLS/组合融合阶段。

原始证据：

- `results/evidence/D15_attribute_expert_seed0/result.json`
- `results/evidence/D16_attribute_expert_seed1/result.json`

## TGAQ 第一阶段

### T0：工程检查

- 零作用路径与冻结 Troika 基线 logits 最大误差为 0；
- 仅 `tgaq.*` 参数可训练，共 11,814,914 个参数；
- 属性文本条件和视觉 cross-attention 均获得有限非零梯度，冻结参数无梯度；
- 非方形 patch 网格会明确报错；
- batch=64 峰值显存约 3.35 GB。

工程门槛通过后，固定同一结构、学习率和损失，分别在 P0/P9 上训练五轮。模型选择
只使用真实对象条件下的 validation 属性 micro accuracy，macro accuracy 作为并列判据。

### T1/T2：双种子正式结果

| 指标 | seed0 / T1 | seed1 / T2 |
|---|---:|---:|
| 最佳轮 | 2 | 4 |
| Troika 原属性头 micro | 34.8164% | 33.2607% |
| TGAQ 局部属性 micro | 32.4207% | 31.4250% |
| micro 变化 | **-2.3958** | **-1.8357** |
| Troika 原属性头 macro | 27.2797% | 30.3970% |
| TGAQ 局部属性 macro | 25.6297% | 23.4279% |
| macro 变化 | **-1.6500** | **-6.9690** |
| 对象 Query 准确率 | 75.6689% | 78.0647% |
| 正确对象属性特异性准确率 | 37.3367% | 37.3367% |
| 正确-错误对象属性 margin | -0.09350 | -0.20627 |
| 目标属性注意力归一化熵 | 0.86329 | 0.85050 |

两个种子均未达到“比 Troika 原属性头至少提高 1 个百分点”的预注册门槛，因此不做
CLS/组合融合、不解冻主干、不读取 test。

训练损失在五轮内持续下降，但 validation 最优分别出现在第 2/4 轮，随后出现退化。
更关键的是，对象 Query 已有 75%-78% 的准确率，正确对象上的真实属性分数却低于
最强错误对象，且训练越久 margin 越负。这说明当前失败不是简单的“对象没有找到”，
而是对象区域约束和属性证据之间没有建立正确对应。

类别级结果并非所有属性都下降。seed0 对 Canvas、Rubber、Sheepskin 有改善，seed1
对 Leather、Satin、Nubuck、Patent Leather 有改善；但提升集合跨种子不一致，并伴随
Nylon、Suede、Synthetic 等属性明显下降。当前类别平衡损失仍没有解决长尾属性的
稳定判别。

原始结果：

- `experiments/T1_tgaq_attr_seed0/result.json`
- `experiments/T2_tgaq_attr_seed1/result.json`
- `experiments/T1_tgaq_attr_seed0/history.json`
- `experiments/T2_tgaq_attr_seed1/history.json`

### D17/D18：固定 Checkpoint 机制消融

固定 T1/T2 最佳 checkpoint，M0 对正式结果的复现误差均为 0。本诊断只读取
validation。

| 模式 | seed0 micro/macro | seed1 micro/macro |
|---|---:|---:|
| Troika 原属性头 | 34.8164 / 27.2797 | 33.2607 / 30.3970 |
| M0 对象上下文 + 正确区域 | 32.4207 / 25.6297 | 31.4250 / 23.4279 |
| M1 对象上下文 + 无区域 | 31.1761 / 26.6904 | 30.4294 / 26.1254 |
| M2 对象上下文 + 错误对象区域 | 32.6385 / 25.8094 | 31.5495 / 22.8191 |
| M3 对象上下文 + 跨图区域 | 32.7318 / 26.2727 | 31.3939 / 24.2536 |
| M4 仅属性文本 + 正确区域 | **39.3279 / 11.9253** | **34.5675 / 10.6405** |
| M5 仅属性文本 + 无区域 | 19.4462 / 9.2239 | 23.7399 / 7.1056 |
| layer6 原始 patch mean | 10.4543 / 5.6297 | 15.0591 / 4.5109 |
| layer6 原始 patch max | 13.0678 / 7.1338 | 17.4860 / 6.8798 |

M2/M3 与 M0 持平或略高，证明当前“正确对象区域”没有对象语义特异性。M1 去掉区域
后两个种子的 micro 下降约 1 点，但 macro 反而提高 1.06/2.70 点，说明区域偏置更像
显著性或类别频率先验，而不是可靠的对象定位。

M4 的 micro 分别比 Troika 高 4.5115/1.3068 点，但 macro 比 Troika 低
15.3544/19.7564 点。类别结果显示它主要预测 Leather、Synthetic 和少数 Rubber，
不是可直接使用的属性专家。M5 又明显下降，说明局部区域确实携带视觉信号，只是该
信号目前严重失衡。layer6 原始 patch mean/max 更低，说明直接 CLIP patch-text
相似度不足，已训练 cross-attention/MLP 提供了真实增益。

因此当前结论不是“patch 无效”，而是：**patch 有属性信号，学习的属性 Query 能提取
一部分信号；但对象 mask 不具语义特异性，属性-only结果又被训练类别先验支配。**
下一步先固定 checkpoint 做训练先验校准与 CLS/局部分数融合诊断；只有双种子同时
改善 CZSL validation AUC，并且不进一步损害属性 macro，才进入下一版重训。

原始证据：

- `results/evidence/D17_tgaq_mechanism_seed0/result.json`
- `results/evidence/D18_tgaq_mechanism_seed1/result.json`
- `docs/TGAQ_MECHANISM_DIAGNOSTIC_PLAN_20260917.md`

## D19/D20：类别先验校准与 CLS 融合

固定 M4 checkpoint，使用训练集属性频率做 `z - tau*log(prior)` 校准，并以
`alpha=0.05/0.10/0.20`加入 Troika。纯 CLS AUC 在两个种子上均以 0 误差复现。

| seed | CLS AUC | 各自最佳 | 最佳 AUC | 变化 |
|---:|---:|---|---:|---:|
| 0 | 50.5867 | pair-only, tau=0, alpha=0.05 | 51.0895 | +0.5028 |
| 1 | 51.4583 | pair-only, tau=1, alpha=0.05 | 51.5721 | +0.1138 |

预注册要求两个种子选中完全相同配置，因此严格门槛失败，不读取 test。共同配置
`pair-only, tau=0, alpha=0.05`在 seed0/1 上分别提高 0.5028/0.1000 个百分点，说明
局部 Query 对组合排序有小幅、跨种子的互补信号；但两个种子的最优校准强度不同。

把局部结果加入 primitive 属性分支没有稳定优势。更重要的是，单独对 M4 做先验校准
并未修复长尾：seed0 的 local micro/macro 从39.3279/11.9253降到 tau=1 时的
16.7393/8.1280；seed1从34.5675/10.6405降到4.6049/3.5311。说明高频塌缩不是一个
简单的后验 prior 偏置，当前局部表示本身缺少稳定的类间结构。

结论：保留“局部分数作为小的 composition residual”这一信号，但关闭当前 TGAQ
属性专家和后验校准路线。不训练融合 gate、不读取 test。下一版改为无对象 mask 的
多层属性表征：第12层提供语义 key、第6层提供细节 value，使用 sparsemax 直接形成
属性选择，并继续保留 CLS 原分支。

原始证据：

- `results/evidence/D19_tgaq_calibration_fusion_seed0/result.json`
- `results/evidence/D20_tgaq_calibration_fusion_seed1/result.json`
- `docs/TGAQ_CALIBRATION_FUSION_PLAN_20260917.md`

## MSAQ V3：无对象 Mask 的多层稀疏属性 Query

MSAQ 使用属性文本 Query；第6层 patch 提供细节 value，第12层 patch 提供语义 key，
通过 sparsemax 选择 patch，不指定 top-k。对象信息不参与 patch 定位。三个 CLS 分支
完全保留，MSAQ 只通过零初始化的 pair/attribute gate 接入。

### M0/M0c 工程检查

真实 batch smoke 中，零 gate 与 Troika logits 最大误差为 0，注意力归一化误差
2.38e-7，只有 MSAQ 与两个 gate 共 5,906,693 参数有梯度，峰值显存 1.95GB。
seed0 一轮后稀疏率从 0.12% 提升到 30.42%，有效 patch 数从 218.29 降到 111.98；
机制不是全局平均，允许进行固定双种子验证。

### M1/M2：双种子五轮验证

| 指标 | seed0 / M1 | seed1 / M2 |
|---|---:|---:|
| 最佳轮 | 3 | 5 |
| CLS AUC | 50.5867 | 51.4583 |
| MSAQ AUC | 51.2964 | 51.5287 |
| AUC变化 | **+0.7096** | **+0.0704** |
| primitive macro变化 | -0.2518 | +0.0325 |
| 局部属性 micro/macro | 26.5090 / 28.5217 | 31.9851 / 31.5611 |
| 稀疏率 | 37.3780% | 53.5938% |
| 有效 patch 数 | 96.52 | 76.89 |

seed0 同时通过 AUC 和 macro 门槛；seed1 的 macro 通过，但 AUC 未达到预注册的
+0.2 点。因此 MSAQ V3 没有通过双种子门槛，不读取 test、不解冻主干、不扫描层位或
稀疏参数。

与 TGAQ 相比，MSAQ 的局部 macro 没有高频塌缩：seed1 达到 31.56%，高于 Troika
primitive macro 30.40%；稀疏率也稳定在 37%-54%。因此“无对象 mask 的属性-first
patch 表征”是有效的。问题在于局部分数没有稳定转化为组合分数。

### D21/D22：MSAQ Gate 机制消融

| 模式 | seed0 AUC变化 | seed1 AUC变化 | 结论 |
|---|---:|---:|---|
| 仅组合残差 | +0.6223 | +0.0573 | 两个 seed 唯一共同正信号 |
| 仅属性残差 | +0.0812 | -0.2113 | 不稳定，seed1有害 |
| 完整 MSAQ | +0.7096 | +0.0704 | seed0 属性残差小幅补充，seed1无效 |

消融证明 MSAQ 的有效信息几乎完全服务于 composition 排序，不是 primitive 属性分支。
代码层面进一步发现当前 `local_pair_logits` 仅按候选的属性 index 从 `(B, A)` 局部
属性 logits 取值：同一属性的所有对象候选得到完全相同的残差。因此它没有提供
attribute-object 特异性，只能做属性级重排序。这解释了 seed0 有信号而 seed1不稳定。

下一版为 MPAQ：属性 Query 仍负责找视觉区域，但最终视觉向量与每个候选 `(属性,
对象)` 的组合文本分别匹配，使对象差异发生在评分端而不引入失败的对象 mask。

原始证据：

- `experiments/M1_msaq_seed0/result.json`
- `experiments/M2_msaq_seed1/result.json`
- `results/evidence/D21_msaq_gate_seed0/result.json`
- `results/evidence/D22_msaq_gate_seed1/result.json`
- `docs/MSAQ_GATE_DIAGNOSTIC_PLAN_20260917.md`

## MPAQ V4 与 MPAQC V5：从全局组合分数到属性内对象残差

MPAQ V4的真实冒烟全部通过，但seed0一轮N0c AUC为50.5823（较CLS -0.0044点），gate
仅-0.0020，因此不进入双seed。D23表明其局部pair分数不能跨属性标定（独立AUC 21.7746%），
但在真实属性已知时对象Top-1为68.4194%、正确对象间隔+0.3992。

据此建立MPAQC V5：局部组合分数在每个属性组内中心化，仅修正同属性对象选择，并改用
条件对象CE。P1c/seed0一轮AUC为50.8415（+0.2548点），primitive macro不变；D24仍有
66.7704%的条件对象Top-1和+0.4850间隔。阶段一通过，现按固定配置运行P1/P2五轮
validation-only；不读取test。完整可复现实验记录见`results/MPAQC_V5_STAGE_20260917.md`。

### MPAQC正式双seed验收

P1/seed0五轮最佳第4轮，AUC50.8647，较P0+0.2780点；P2/seed1最佳第1轮，AUC51.4671，
较P9仅+0.0089点。两seed primitive macro均完全保持，但P2低于+0.2点门槛，故MPAQC V5
不通过、未读取test、未进行同版超参数扫描。下一步仅做固定checkpoint的条件对象排序
互补性诊断，详见`results/MPAQC_V5_STAGE_20260917.md`。
