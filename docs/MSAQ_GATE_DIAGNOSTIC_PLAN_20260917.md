# MSAQ 固定 Checkpoint Gate 机制消融

日期：2026-09-17

## 目的

M1/M2中，MSAQ局部属性 macro 已显著优于此前 TGAQ，但 CZSL AUC 在 seed0 与 seed1
不稳定。固定两个最佳 checkpoint，拆开两个残差 gate，判断局部证据主要服务于组合
排序还是 primitive 属性分支。

## 对照

1. 两个 gate 都关闭：严格 CLS；
2. 仅组合残差：只给 `(attribute, object)` pair 分数加 MSAQ 局部属性分数；
3. 仅属性残差：只给 primitive attribute 分支加 MSAQ 局部属性分数；
4. 完整 MSAQ：两个已学习 gate 均打开。

所有模式使用同一 checkpoint、同一 validation 顺序。只评估 validation，不扫描 gate
强度，不读取 test。

## 判定

- 若仅组合残差在两个 seed 都不低于完整模型：MSAQ 的有效信息主要是组合排序证据；
- 若仅属性残差在两个 seed 都改善 primitive macro：局部属性专家可独立保留；
- 若 seed1 的所有模式都近乎 CLS：当前数据/种子下的局部证据没有稳定进入 CZSL；
- 该诊断不允许触发任何 test 或超参数搜索。
