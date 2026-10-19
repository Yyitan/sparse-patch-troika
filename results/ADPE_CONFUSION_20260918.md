# ADPE 属性混淆审计结果：D39 / D40

日期：2026-09-18。实验为 UT-Zappos validation-only，零训练，未读取 test 图像。

## 1. 目的

检查当前 ICAQ 的局部属性分数是否能处理 Troika 最主要的 hard negative：对象已经正确，
但属性被预测成相似的错误属性。审计时强制 `msaq_pair_gate=0`，ICAQ 不改变 Troika
组合预测。

## 2. 结果汇总

| 指标 | D39 / seed0 | D40 / seed1 |
|---|---:|---:|
| Troika 属性错、对象对 / 全部样本 | 55.5383% | 49.3777% |
| Troika 属性错、对象对 / 组合错误 | 79.8658% | 75.2846% |
| ICAQ 属性 Top-1 | 31.2694% | 30.6161% |
| ICAQ 属性 Top-5 | 78.9981% | 74.6422% |
| ICAQ真实属性 - Troika错误属性 margin均值 | -3.6933 | -3.4253 |
| ICAQ正 margin / hard error | 6.0504% | 5.7341% |
| ICAQ正 margin / 全部样本 | 3.3603% | 2.8314% |
| 预注册是否通过 | 否 | 否 |

ICAQ 的 margin 比 Troika 原属性分支更好：Troika 原属性分支在 hard error 上的 margin
均值为 `-6.6469/-6.4596`。但 ICAQ 仍然很少能把真实属性排在 Troika 错误属性前面，
说明它只有弱互补信号。

## 3. 主要属性混淆

两个 seed 的 Troika 错误结构高度一致，主要集中在材质属性：

- `Synthetic -> Leather`：seed0 569 个，seed1 432 个；
- `Full.grain.leather -> Leather`：seed0 392 个，seed1 390 个；
- `Rubber -> Leather`：seed0 163 个，seed1 128 个；
- `Faux.Leather -> Leather`：seed0 114 个，seed1 108 个；
- `Patent.Leather -> Leather`：seed0 103 个，seed1 106 个。

这说明当前模型不是完全没有材质信息，而是存在明显的 Leather 高频先验塌缩。

## 4. 预注册判断

ADPE 的预注册门槛要求两个 seed 同时满足：

1. hard error 中正 margin 至少 10%；
2. 全部样本中正 margin 至少 5%；
3. ICAQ 属性 Top-5 至少 70%。

两个 seed 只通过了第 3 项，未通过前两项。因此不启动当前 ADPE 训练，也不继续扫描
ICAQ 的 patch 数量、gate、融合权重或 Query 数量。

## 5. 结论与下一步

当前 ICAQ 的弱信号说明 patch 属性证据并非完全无效，但它不能稳定解决最关键的材质
混淆。继续在同一 CLIP patch 特征上训练 Query，可能仍然被 Leather 高频先验吸收。

下一步转向互补视觉骨干可行性验证：先用冻结的 DINO/EVA 等视觉 patch 特征做属性线性
探针或最近邻审计，只回答它们能否纠正 `Synthetic/Full.grain.leather/Rubber/Faux.Leather
-> Leather` 这类错误；不马上接入完整 CZSL 模型。只有互补性在两个 seed 上成立，才把
它和 CLIP 的 CLS/patch 特征融合。

