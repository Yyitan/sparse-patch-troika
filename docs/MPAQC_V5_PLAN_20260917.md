# MPAQC V5：属性内条件对象残差

日期：2026-09-17

## 由 MPAQ V4 得到的证据

MPAQ V4 让属性局部视觉向量与每个组合文本直接匹配，解决了 MSAQ 对同一属性不同对象
给出相同 residual 的结构问题。真实 checkpoint 冒烟确认零 gate 与 CLS 完全一致，且同属性
不同对象的初始分数有非零方差。

但 seed0 一轮后，整体组合 AUC 为 50.5823，较 CLS 的 50.5867 低 0.0044 点，门控只有
-0.0020，不允许直接进入双 seed 五轮。D23 固定该 checkpoint 后发现：

| 指标 | 数值 |
|---|---:|
| 局部组合 scorer 单独 AUC | 21.7746 |
| 局部 scorer 全 pair Top-1 | 15.9303% |
| 已知真实属性时的对象 Top-1 | **68.4194%** |
| 正确对象相对最强同属性错误对象间隔 | **+0.3992** |
| 每个属性平均候选对象数 | 10.1721 |

因此局部组合分数不能做跨属性全局排序，但包含明显的**属性条件对象辨别**信号。V4 的全局
pair CE 强迫它同时承担属性与对象标定，与其实际能力不匹配。

## V5 结构

1. 保留 MPAQ 的属性文本 sparsemax patch 定位，第12层语义 key + 第6层细节 value；
2. 仍计算原始局部组合分数 `s(a,o)`；
3. 对每一个属性的所有候选对象做均值中心化：
   `r(a,o)=s(a,o)-mean_o'(s(a,o'))`；
4. 只将 `r(a,o)` 通过零初始化 composition gate 加到 CLS composition logits；
5. 每个属性组的 residual 均值严格为零，故它不应直接改变 CLS 的属性级排序，只校正
   同属性下的对象选择；
6. primitive attribute 与 object CLS 分支均不修改。

## V5 训练

冻结 Troika，只训练 MSAQ/MPAQ 局部分支及 composition gate。

- 原 Troika 全组合 CE：使 residual 为最终 CZSL 排序服务；
- **条件局部 pair CE**：仅在真实属性对应的对象候选之间分类，不再训练局部分数跨属性竞争；
- 原有局部属性标准 CE + Balanced Softmax CE：维持 patch 属性定位与长尾结构；
- 不训练属性 residual，不使用对象 mask，不读取 test。

## 检查与决策

工程检查：零 gate 精确复现 CLS；sparsemax 合法；原始 pair score 与中心化 residual 方差
均非零；任一属性组 residual 均值为零。

阶段一只运行 seed0 一轮。只有同时满足以下条件，才启动固定五轮双 seed：

1. 稀疏注意力非退化，且条件对象诊断 Top-1 不低于 D23 的 60%；
2. 验证 AUC 不出现明显下降（低于 CLS 不超过 0.2 点）；
3. primitive attribute macro 下降不超过 0.5 点。

正式门槛：两个 seed 的 validation AUC 都至少高于各自 CLS 0.2 点，且两个 seed 的
primitive macro 下降均不超过 0.5 点。任一失败，不读 test，不做额外超参数扫描。

## 阶段一结果：P1c 与 D24

P1c 使用 P0/seed0 checkpoint 训练一轮。AUC 从 50.5867 升至 **50.8415**，变化
**+0.2548 点**；primitive macro 保持 27.2797%，稀疏率 28.7896%，有效 patch 数
115.88，composition gate 为正的 0.00709。三项阶段一门槛均通过。

D24 固定 P1c 权重只读 validation：局部 scorer 的跨属性 AUC 为17.0976%，但真实属性
条件下对象 Top-1 为 **66.7704%**，正确对象相对同属性最强错误对象间隔为 **+0.4850**，
高于预注册的60%条件门槛。由此批准固定配置的双 seed 五轮验证；不改变损失权重、层位、
稀疏参数或 gate 初始化。

## 正式结果：P1/P2 五轮双 seed 验证

| 指标 | seed0 / P1 | seed1 / P2 |
|---|---:|---:|
| 最佳轮 | 4 | 1 |
| CLS AUC | 50.5867 | 51.4583 |
| MPAQC AUC | **50.8647** | 51.4671 |
| AUC变化 | **+0.2780点** | **+0.0089点** |
| primitive macro变化 | 0.0000点 | 0.0000点 |
| 局部属性 micro/macro | 27.5980 / 31.9754 | 28.4381 / 19.1487 |
| 稀疏率 | 45.4131% | 43.8939% |
| 有效 patch 数 | 90.06 | 95.54 |
| composition gate | +0.03047 | -0.00412 |

P1通过两个正式门槛；P2的 macro 通过，但 AUC 仅提高0.0089点，低于预注册的0.2点。
故 V5 未通过双 seed 验收：不读取 test，不继续训练同版，不扫描层位、损失权重或 residual
尺度。保留的事实是，属性-first 稀疏 patch 表征与条件对象区分确实可学习；未解决的是其
对两个 seed 的 CLS 组合排序是否具有一致的互补性。
