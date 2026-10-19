# MPAQC：CLS与局部条件对象排序互补性诊断

日期：2026-09-17。使用已经完成的 P1/P2 最佳 checkpoint，仅 validation，不训练、不调参、
不读取 test。

## 问题

MPAQC 局部 scorer 在真实属性已知时具有约67%的对象Top-1，但P2的全局CZSL AUC几乎未提升，
且其学习gate为负。需要区分两种解释：

1. CLS在真实属性条件下已经比局部 scorer 更强，局部没有可利用的对象错误余量；
2. 局部与CLS各自正确的样本互补性不足，或局部在CLS最容易错的样本上更差；
3. 局部有互补正确样本，但固定全局scalar gate无法只在这些样本上启用。

## 固定比较

在每张验证图的真实属性对应候选对象集合内，分别计算：

- 冻结 CLS composition logits 的对象Top-1和正确对象margin；
- raw local pair scorer 的对象Top-1和margin；
- 已训练 MPAQC 融合 logits 的对象Top-1和margin；
- CLS错且局部对、CLS对且局部错、二者均对/均错的比例；
- “二者任一正确”的oracle条件对象Top-1上界。

## 决策

- 若CLS错/局部对比例很低，停止局部对象 residual 路线；
- 若该比例明显而完整融合仍无改善，下一版仅考虑由冻结CLS置信度驱动的样本级gate，且先做
  双seed固定checkpoint上界诊断；
- 无论结果如何，不读取test。
