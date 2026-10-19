# 真实前景与 MSAQ 属性注意力对齐诊断

日期：2026-09-17。仅使用M1/M2已完成最佳权重和validation；不训练、不读取test。

## 前提

F0证明UT-Zappos白背景最大连通域可稳定得到鞋主体。仍不能直接假设强制前景会有效：MSAQ
可能已经自然聚焦前景，强制mask反而丢失边缘属性信息。

## 固定测量

按验证预处理 `Resize(224)+CenterCrop(224)` 映射前景mask至ViT-L/14的16×16 patch，计算真实属性对应
attention的前景质量，并与该图的均匀patch前景覆盖比较：

- 真实属性attention落在foreground的质量；
- 所有属性平均attention的前景质量；
- 均匀attention的前景质量；
- 真实属性attention相对均匀的富集倍数；
- 最大注意力patch的前景核心命中率。

## 决策

- 若两个seed真实属性attention已显著高于均匀且top patch多数位于foreground core，说明MSAQ
  已使用真实主体，禁止仅靠硬mask重复训练；
- 若注意力接近均匀或大量落在背景，才允许FGAQ中将foreground作为attention合法域，并先做
  单seed一轮机制检查；
- 不读取test。

## D29/D30 结果：已有MSAQ已经强前景对齐

| 指标 | M1 / seed0 | M2 / seed1 |
|---|---:|---:|
| 真实属性attention前景质量 | 76.0739% | 94.5498% |
| 所有属性attention前景质量 | 78.3959% | 93.7810% |
| 均匀patch前景质量 | 44.4528% | 44.4528% |
| 真实属性相对均匀富集 | 1.7874× | 2.2332× |
| 最大注意力patch前景核心命中 | 98.6310% | 99.5644% |

两个seed均显著超过均匀foreground覆盖，且最大属性注意力patch几乎总在真实鞋主体核心内。
故对象/前景定位不是MSAQ当前瓶颈；不训练foreground硬mask、core/ring或中心邻域变体，避免
对已完成的空间选择做冗余约束。后续若继续视觉端，应改进前景patch的属性判别表征和它与CLS
composition分数的兼容性，而非继续优化空间mask。
