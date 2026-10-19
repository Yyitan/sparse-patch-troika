# ICAQA V8：最后两层视觉Adapter与CLS蒸馏

日期：2026-09-17。独立代码目录`icaqa_troika`；原Troika、CAMS、ICAQ目录均不修改。
全部阶段validation-only，双seed正式门槛通过前禁止读取test。

## 1. 进入依据

ICAQ在两个seed的局部属性macro达到27.96%/32.21%，错配图像后仅保留21.57%/22.45%，
证明patch属性证据真实；但composition AUC为+0.1690/-0.1244点，冻结视觉端的局部特征与
原CLS组合空间不够兼容。继续扫gate或patch层位不能解决表征空间不匹配。

## 2. 可训练结构

保留ICAQ的单图像Query、layer12语义key、layer6细节value和sparsemax。ViT-L/14共有24个
block，每个block含attention/FFN两个Troika Adapter。只解冻最后两个block对应的四个Adapter：
索引22、23、46、47；CLIP主干、文本端、CMT、disentangler和其余Adapter全部冻结。

可训练参数为ICAQ、composition gate和上述四个Adapter。ICAQ学习率1e-4，Adapter学习率
1e-5，AdamW weight decay 1e-4；不修改patch层位、头数或损失权重。

## 3. 在线CLS教师约束

训练增强包含随机翻转、透视和旋转，不能使用离线缓存教师。每个batch用同一增强图像同时
输入冻结Troika教师与学生。学生在加入局部composition residual前保存三个CLS logits：
composition、attribute、object；分别对教师logits做温度2.0的KL蒸馏，三项平均，权重1.0。

总损失：composition CE 1.0 + ICAQ标准/平衡属性CE各0.5合并 + CLS蒸馏1.0。蒸馏不约束
局部residual本身，只限制Adapter不要破坏原CLS能力。

## 4. 验证顺序

### C0：工程检查

验证教师/学生初始CLS和zero-gate输出误差小于3e-4；只有ICAQ、gate和四个目标Adapter有
梯度；教师无梯度；蒸馏相同logits为零；峰值显存可在单卡安全运行。

### C0c：seed0一轮

进入正式双seed条件：AUC相对基线下降不超过0.2点；primitive macro下降不超过0.5点；
局部macro至少25%；类别覆盖至少12/16；CLS蒸馏有限且训练稳定。

### C1/C2：固定五轮双seed

仅C0c通过后启动。两个seed均要求validation AUC至少高于各自CLS 0.2点，primitive macro
下降不超过0.5点，局部macro至少25%。任一失败即关闭V8，不读取test、不调蒸馏权重、
Adapter层数、学习率或gate。

## 5. 最终判定

若V8通过双seed，才允许一次test评估；若失败，说明当前属性patch证据虽然真实，但不能通过
属性级加性residual解决CZSL组合排序，后续应转向显式关系建模或训练目标重构，而不是继续
扩大视觉端可训练范围。
