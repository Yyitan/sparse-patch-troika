# CAMS Attribute Latent 分工上界诊断方案

日期：2026-09-17

## 问题

Stage 9证明对象prompt对32个CAMS latent units的后验权重几乎均匀，不能作为选择器。
但这仍不能区分两种情况：各unit确实没有分工，或unit存在互补属性证据但对象文本无法
读取。本阶段只验证attribute units自身是否存在可利用的分工上界。

## 固定条件

- 使用Stage 9相同的seed0/seed1原复现checkpoint。
- closed-world validation，与CAMS checkpoint选择口径一致。
- 不读取test，不训练参数，不扫描温度、unit数量或融合系数。
- global、composition和object分支完全固定，只替换attribute logits的聚合方式。

## 候选与控制

| 名称 | 定义 |
|---|---|
| mean_feature | 原CAMS先平均attribute units再投影，严格基线 |
| unit_mean | 每个unit独立投影并计算logits后平均 |
| attribute_max | 每个候选属性用其prompt选择响应最大的unit |
| attribute_logmeanexp | 对每个候选属性在units上做无参数log-mean-exp |
| shuffled_image_max | 对完整validation做固定随机置换后使用另一张图的units，检验图像特异性 |

另记录三个不用于推理的上界：任一unit预测正确、按真实属性target logit选择unit、
最佳单个unit；同时统计每张图32个units预测出的不同属性数量和unit特征两两余弦。

## 门控

每个seed满足以下任一条件才保留latent分工路线：

1. `attribute_max`或`attribute_logmeanexp`同时在pair AUC和局部属性准确率上超过
   `mean_feature`及`shuffled_image_max`；
2. 任一unit正确率相对mean至少高5个百分点，真实属性target选择准确率相对mean至少
   高3个百分点，且每张图平均至少出现2种不同unit属性预测。

两个seed都必须通过同类门控，才能继续增强一致性或训练属性专用router。否则停止
patch/latent选择路线。

## 输出

- `diagnostics/L2_cams_attr_unit_specialization_seed0_globalshuffle_val/`
- `diagnostics/L3_cams_attr_unit_specialization_seed1_globalshuffle_val/`
- 汇总：`results/STAGE10_20260917.md`

首次L0/L1使用batch内相邻样本roll。由于validation按类别有序，该控制存在标签相关，
保留为审计记录但不用于正式门控；L2/L3改为全验证集、固定随机种子的图像置换。
