# CAMS Latent Object-Conditioned Attribute 诊断方案

日期：2026-09-17  
独立代码：`cams_latent_v1/`  
原始CAMS只读目录：`./lyt/czsl/CAMS-main/CAMS`

## 动机

Stage 8证明第6层patch包含属性信息，但原始CLIP对象文本与patch余弦相似度形成的
mask缺少对象语义特异性。CAMS已经使用32个可学习latent units通过Gated
Cross-Attention聚合全部视觉token，因此下一步先判断对象语义能否在latent空间中
选择出更可靠的属性证据。

## 固定Checkpoint

- CAMS UT-Zappos seed0：`./CAMS/output/utzappos_crdiff/val_best.pt`。
- CAMS UT-Zappos seed1：`./CAMS/output/utzappos_seed1/val_best.pt`。
- `<server>/output/ut_seed0`和`ut_seed1`当前只保留训练日志，没有checkpoint。
- 已核验`./CAMS`与指定的`<server>/CAMS`：
  `model/cams.py`、`clip_modules/model.py`及seed0配置的SHA256逐项一致，因此使用
  前者保留的原复现checkpoint、后者的只读代码和配置。
- 两个原始CAMS目录和checkpoint均只读，所有新增代码和结果写入本项目。
- 只评估validation，不读取test，不重新训练CAMS。validation显式使用closed-world，
  与CAMS `train.py`选择`val_best.pt`时的数据集构造方式保持一致；配置中的
  `open_world: true`只属于最终测试设置，不能传给validation。

## 方法

从CAMS视觉编码器取得GCA后的32个latent units，分别通过已有`tr_a`和`tr_o`
得到属性与对象latent序列。对象latent与对象prompt计算相似度，并在32个units上做
temperature=0.07的softmax。该权重聚合属性latent，再使用原有`ln_norm`和
`proj_att`得到对象条件化属性表示。

uniform权重应精确复现原CAMS属性分支，因此它同时是实现正确性检查和baseline。

## 控制组

| 名称 | 定义 |
|---|---|
| uniform | 32个latent units均匀平均，复现CAMS原属性分支 |
| correct_object | 当前候选对象prompt产生的latent权重 |
| wrong_object | 对象索引确定性错位后的latent权重 |
| shuffled_mask | 使用另一张图的同对象latent权重 |

候选pair评分保持CAMS的global/composition/object部分不变，仅将原属性概率替换为
对象条件化属性概率。真实对象标签只用于独立的局部属性准确率诊断。

## 门控

每个seed必须同时满足：

1. uniform与原CAMS pair logits最大误差小于1e-5；
2. correct_object validation AUC高于uniform、wrong_object、shuffled_mask；
3. correct_object局部属性准确率高于以上三个控制组。

只有seed0和seed1均通过，才复制CAMS进入可训练的条件属性latent head；否则停止
对象条件化选择，转向属性专用多latent或增强一致性，而不投入完整训练。

## 记录位置

- `diagnostics/C2_cams_latent_controls_seed0_closed_val/`
- `diagnostics/C3_cams_latent_controls_seed1_closed_val/`
- 汇总：`results/STAGE9_20260917.md`

首次运行的C0/C1错误地把`open_world: true`用于validation，保留其JSON作为协议审计
记录，但不用于方法门控或结果汇总。
