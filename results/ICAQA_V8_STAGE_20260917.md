# ICAQA V8 最后两层视觉Adapter阶段结果

日期：2026-09-17。全部实验为UT-Zappos validation-only，未读取test。

## 1. 验证目标

V7已证明双seed的ICAQ patch属性证据真实，但直接composition residual不稳定。V8仅解冻
ViT-L/14最后两个block的四个Troika Adapter（索引22、23、46、47），并以冻结Troika
教师在线蒸馏学生加局部residual前的composition/attribute/object CLS logits。

固定配置：ICAQ学习率1e-4，Adapter学习率1e-5，蒸馏温度2.0、权重1.0；其余视觉主干、
文本端、CMT、disentangler和前22层Adapter全部冻结。

## 2. C0工程检查

两个蒸馏单测通过。首次smoke发现教师已切换eval、学生缓存文本时仍为train，文本/CMT
dropout导致初始CLS误差4.463；这是工程状态不一致，不是实验结果。修复为教师和学生均先
eval再缓存文本，未修改任何超参数。

修复后C0结果：

- zero gate相对教师最大误差：0；
- 三个原始CLS logits相对教师最大误差：0；
- attention归一误差：1.19e-7；
- 蒸馏相同logits损失：约-1.4e-9（数值零）；
- 只有ICAQ、composition gate和Adapter 22/23/46/47有梯度；
- 可训练参数：6,435,332；
- smoke峰值显存：约3.69GiB。

## 3. C0c seed0一轮

| 指标 | 训练前 | 一轮后 | 变化 |
|---|---:|---:|---:|
| validation AUC | 50.5867 | 50.3532 | -0.2336点 |
| primitive attribute macro | 27.2797% | 27.1329% | -0.1468点 |
| primitive attribute micro | 34.8164% | 34.8787% | +0.0622点 |
| 局部属性macro | 7.1429% | 31.3210% | +24.1782点 |
| 局部属性micro | 0.9956% | 31.4872% | +30.4916点 |
| 局部类别覆盖 | 2/16 | 13/16 | +11类 |
| attention稀疏率 | 0.1157% | 2.6477% | +2.5320点 |

训练轮均CLS蒸馏损失为0.002409，说明教师约束有效；但AUC下降0.2336点，超过预注册的
“最多下降0.2点”门槛。训练峰值显存约7.11GiB。

## 4. 决策

C0c未通过，取消C1/C2双seed，不读取test，不调蒸馏权重、Adapter学习率、解冻层数或gate。
V8说明问题并非简单的视觉端容量不足：局部属性识别持续有效，CLS也大致保持，但属性级
加性residual仍改变了组合排序并降低AUC。

至此关闭“继续扩大视觉端解冻范围”路线。下一阶段若继续，应把局部属性证据用于显式的
属性-对象关系建模、候选重排序训练目标或组合级对比约束，而不是继续选择patch、放大gate
或解冻更多ViT层。新方向需先讨论并单独预注册。
