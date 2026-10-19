# TGAQ 类别先验校准与 CLS 融合方案

日期：2026-09-17

## 背景

D17/D18 显示，去掉对象上下文、保留局部区域的属性文本 Query（M4）在两个种子上
均提高属性 micro，但 macro 严重下降，预测集中于高频属性。该结果可能来自两种情况：

1. 局部属性表示有用，只是训练类别先验导致分数失准；
2. 提升完全依赖高频类别，无法为 CZSL 组合排序提供互补证据。

本轮固定 T1/T2 最佳 checkpoint，不重新训练，只在 validation 上区分这两种情况。

## 固定候选

使用训练集属性频率 `pi(a)` 对 M4 局部属性 logits 做后验校准：

`z_cal(a) = z_local(a) - tau * log(pi(a))`

- `tau = 0, 0.5, 1.0`；
- 残差系数 `alpha = 0.05, 0.10, 0.20`；
- `pair_only`：只向 Troika composition logits 加候选 `(attribute, object)` 的局部分数；
- `pair_attr`：除 composition 外，还按 Troika 对象后验聚合局部属性 logits，加入
  primitive attribute 分支；
- 纯 Troika CLS 作为可选基线。

所有候选共享一次视觉前向。没有层数、patch 数、邻域、温度或其他损失扫描。

## 门槛

1. 纯 CLS 必须精确复现对应 P0/P9 validation AUC；
2. 候选的 primitive 属性 macro 最多比 CLS 下降 0.5 个百分点；
3. 在满足 2 的候选中按 validation AUC 选模；
4. 两个种子必须选中完全相同的 `mode/tau/alpha`，且各自 AUC 至少提高 0.1 个百分点；
5. 任一条件失败，则不读取 test、不训练融合 gate，并转向无对象 mask 的多层属性表征。

本轮结果只能证明固定 checkpoint 的互补性，不能直接作为最终方法结果。
