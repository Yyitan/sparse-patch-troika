# Patch 对象定位机制控制实验方案

日期：2026-09-17  
代码版本：`code_v8/`  
数据集：UT-Zappos closed-world CZSL

## 目标

本阶段不训练新模型，只回答一个机制问题：第12层对象文本产生的 soft patch mask
是否真的比无对象信息的空间先验更有利于第6层属性识别。此前固定 checkpoint 的
组合 AUC 小幅提升不足以证明对象定位，因为 soft map 归一化熵约为0.989，接近均匀。

## 固定设置

- P0 seed0 与 P9 seed1 的最佳 CLS checkpoint。
- object layer=12，attribute layer=6。
- softmax temperature=0.07，composition residual alpha=0.10。
- 只评估 validation，不读取 test。
- 两个 seed 独立运行，不根据单个 seed 修改参数。

## 控制组

| 名称 | 定义 | 检验内容 |
|---|---|---|
| uniform | 所有 patch 均匀权重 | 增益是否只是 patch 平均 |
| center | 固定中心高斯先验 | 增益是否只是鞋类图像的中心偏置 |
| correct_object | 当前候选对象文本的 soft mask | 目标机制 |
| wrong_object | 确定性错位对象文本 mask | mask 是否依赖正确对象语义 |
| shuffled_mask | 同 batch 另一张图的同对象 mask | mask 是否依赖当前图像 |

## 指标

1. 保持原 CLS attr/obj 分支，只向 composition 加入局部属性项后的验证 AUC。
2. 使用真实对象索引选择 mask 后，纯局部属性 logits 的 top-1 准确率。
3. 真实属性 logit 均值和 mask 归一化熵。
4. correct_object 相对每个控制组的 AUC 与局部属性准确率差值。

真实对象索引只用于机制诊断中的局部属性准确率，不用于最终候选 pair 评分，也不
形成可部署方法。

## 预注册门控

每个 seed 中，correct_object 必须同时满足：

- pair AUC 高于 uniform、center、wrong_object、shuffled_mask；
- 局部属性准确率高于以上四个控制组。

只有两个 seed 都通过，才进入“冻结/stop-gradient mask + 条件属性 head + 小门控”
训练。任一 seed 失败则停止当前 CLIP 相似度 mask 路线，下一步转向 CAMS latent
cross-attention，不继续扫描 layer、top-k、邻域、temperature 或 alpha。

## 结果记录

- `diagnostics/D8_objmask_controls_seed0_val/`
- `diagnostics/D9_objmask_controls_seed1_val/`
- 汇总报告：`results/STAGE8_20260917.md`

## 完成结果

D8/D9均未通过预注册门控。seed0的correct-object局部属性准确率高于全部控制组，
但pair AUC低于center；seed1的correct-object pair AUC高于其他patch控制组，但局部
属性准确率低于shuffled mask。完整数据与解释见`results/STAGE8_20260917.md`。

因此停止当前CLIP对象相似度mask路线，不运行条件属性head训练，下一阶段按计划转向
CAMS latent cross-attention诊断。
