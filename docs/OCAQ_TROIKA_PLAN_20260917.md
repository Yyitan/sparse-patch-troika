# OCAQ：对象条件属性 Query 视觉分支验证方案

日期：2026-09-17

## 1. 研究问题

当前实验已经否定了两类简单做法：

1. 直接删除、max 或 top-k 聚合 CLIP patch，不能稳定超过 Troika 的 CLS 基线。
2. 用原始“对象文本-视觉 patch 相似度”形成固定 mask，再池化属性 patch，冻结诊断虽偶有增益，但端到端训练和对象特异性对照均失败。

这并不等于 patch 没有价值。现有结果更支持下面的判断：CLS 负责稳定的全局组合判断，patch 适合补充属性细节，但需要一个受到属性标签监督、能够主动学习分工的提取器，而不是固定相似度规则。

本阶段验证 **Object-Conditioned Attribute Query（OCAQ）**：完整保留 Troika 的 CLS 组合、属性和对象分支，只增加一个从中层 patch 提取属性细节的可训练残差分支。

## 2. 固定基线

- 数据集：UT-Zappos，closed-world validation。
- seed0 基线：`P0_ut_cls_seed0/val_best.pt`，validation AUC 50.5867。
- seed1 基线：`P9_ut_cls_seed1/val_best.pt`，validation AUC 51.4583。
- 两个种子必须分别加载对应 checkpoint，禁止交叉比较。
- 第一阶段冻结整个 Troika 基线，只训练 OCAQ 新增参数。
- 模型选择只看 validation；门控未通过前不运行 test。

## 3. 第一版结构

### 3.1 保留 CLS 主干

原始 Troika 输出保持不变：

- 组合 CLS logits；
- 属性 CLS logits；
- 对象 CLS logits；
- 原始推理融合公式。

OCAQ 只向组合 logits 添加零初始化门控的局部残差。因此训练开始时模型严格等价于原 checkpoint，不会一开始破坏基线。

### 3.2 对象 Query 与空间区域

- 使用 ViT 第 12 层 patch，原因是之前的翻转诊断中该层对象响应更稳定。
- 每个候选对象文本经过投影后作为对象 Query，对第 12 层 patch 做可训练 cross-attention。
- 得到对象上下文和对象空间概率图。
- 对象图不做 hard top-k 删除，只作为后续属性 Query 的软注意力偏置。
- 在 patch 网格上使用固定 3x3 平均扩张，形成“对象本体 + 周围一圈”的区域先验。
- 为避免 4 个 Query 在初始化时立即重合，四者分别从“偏核心、核心与邻域混合、偏邻域、近全局”的可学习偏置开始；每个 Query 的原始图/扩张图混合系数和偏置强度都可训练，不是固定 mask。

### 3.3 属性 Query

- 使用 ViT 第 6 层 patch，原因是之前固定 checkpoint 诊断中该层提供了最强的属性补充信号。
- 每个对象使用 K=4 个可学习属性 Query。
- Query 由“共享 Query 向量 + 对象视觉上下文 + 对象文本”组成。
- Query 对第 6 层 patch 做 cross-attention，并在 attention logits 中加入对象区域的对数软偏置。
- 每个 Query 输出一个局部属性特征，与全部属性文本计算相似度。
- 4 个 Query 的属性证据用 log-sum-exp 聚合，同时保留每个 Query 的注意力图和负载统计用于塌缩约束。

### 3.4 组合残差

对于候选组合 `(属性 a, 对象 o)`，从对象 `o` 对应的局部属性 logits 中取属性 `a` 的分数，形成局部组合残差：

`最终组合 logits = 原 Troika 组合 logits + sigmoid(gate) * local_scale * 局部组合 logits`

同时用原 Troika 的对象 CLS 后验对全部对象条件下的局部属性 logits 加权，形成图像级属性残差：

`最终属性 logits = 原 Troika 属性 logits + tanh(attr_gate) * 对象后验加权的局部属性 logits`

组合和属性 gate 都使用严格零影响初始化；训练初始输出必须与原 Troika 数值一致。对象 CLS logits 始终不修改。

## 4. 训练损失

第一阶段固定以下损失，不做网格扫描：

1. `L_pair`：加入局部残差后的组合交叉熵。
2. `L_attr`：加入局部残差后的属性交叉熵。
3. `L_object_query`：对象 Query 的局部对象分类交叉熵，显式监督对象区域。
4. `L_attr_query`：在真实对象条件下，局部属性预测的交叉熵。
5. `L_specificity`：真实对象的目标属性分数必须高于 batch 内错误对象，使用 margin ranking。
6. `L_diversity`：同一对象的不同 Query 注意力图不能完全重合。
7. `L_balance`：避免长期只有一个 Query 获得全部聚合权重。

OCAQ 辅助项初始固定权重：对象 Query `0.5`、属性 Query `1.0`、对象特异性 `0.5`、多样性 `0.05`、负载均衡 `0.05`；Troika 原始 pair/attr/obj 权重保持不变。本阶段不扫描损失权重。增强一致性先作为第二门控后的增量实验，避免第一版同时引入过多变量和双视图显存开销。

## 5. 实验顺序与门控

### D11：对象 Top-M 召回诊断

在 P0/P9 validation 上统计对象 CLS 分支的 top-1/2/3/5 recall。

- 若两个种子 top-3 recall 都不低于 98%，后续大对象集合可优先只计算 top-3 对象 Query。
- UT-Zappos 只有 12 个对象，第一版仍对全部对象计算，以免候选截断干扰核心结论。

### Q0：工程检查

- batch=2 的直接前向与缓存文本前向一致。
- gate 零影响时，最终 logits 与基线 logits 一致。
- OCAQ 参数梯度非零、有限；冻结基线参数无梯度。
- 记录峰值显存。
- 检查 4 个 Query 的注意力熵、两两重合度和使用率。零初始化时只要求四种空间角色数值不同；再运行 seed0 一轮短训练，若重合度没有下降趋势或有效 Query 数不大于 2，则停止正式双种子训练。

### Q1/Q2：冻结基线、双种子训练

- Q1：seed0 加载 P0。
- Q2：seed1 加载 P9。
- 冻结全部原 Troika 参数，只训练 OCAQ。
- 先训练 5 个 epoch；每轮保存 validation 指标和 Query 诊断。
- 不运行 test。

进入下一阶段的必要条件：

1. 两个种子的最佳 validation AUC 都至少高于对应 CLS 基线 0.3 个百分点。
2. 两个种子的 validation 属性准确率都至少提高 1.0 个百分点。
3. 两个种子中，真实对象条件下的目标属性分数都至少比错误对象高 1.0 个百分点或等价 margin 指标为正。
4. Query 没有塌缩：有效 Query 数大于 2，且平均注意力两两重合度低于 0.9。

### Q3/Q4：解冻 Troika 轻量模块

只有 Q1/Q2 同时通过才运行：

- 解冻 visual adapters、CMT、prompt/context、disentangler 和 OCAQ；CLIP 主体仍冻结。
- 从 Q1/Q2 最佳 checkpoint 继续训练，不从头随机训练。
- 使用更小学习率和基线 logits 蒸馏，减少原能力漂移。
- 仍先只看 validation，双种子通过后才允许固定配置测试一次。

## 6. 结果记录规范

每个实验必须保存：

- 完整配置 JSON；
- 代码 SHA256 和 checkpoint SHA256；
- 训练日志、逐轮 history、progress、失败原因；
- 最佳 validation checkpoint；
- AUC、属性准确率、对象准确率；
- Query 注意力熵、重合度、使用率、真实/错误对象差值；
- 峰值显存和运行时间。

阶段结果统一写入 `results/OCAQ_STAGE_20260917.md`，所有结论使用中文。失败实验保留，不覆盖、不删除。

## 7. 当前允许与禁止事项

允许：按 D11 -> Q0 -> Q1/Q2 -> Q3/Q4 的顺序执行，并在空闲 GPU 上最多并行三个本项目任务。

禁止：在 Q1/Q2 通过前读取测试集；根据测试结果回调结构；扫描层数、Query 数、邻域大小、温度、损失权重或残差系数；改动原始 Troika/CAMS 目录。
