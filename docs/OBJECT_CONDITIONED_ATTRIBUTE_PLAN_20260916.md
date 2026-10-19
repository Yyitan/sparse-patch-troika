# 对象条件化局部属性 Patch 验证方案

日期：2026-09-16  
代码版本：`code_v6/`

## 1. 核心假设

CLS继续承担完整对象、场景和组合语义，patch不替代CLS，只补充更细粒度的属性
证据。属性通常依附于对象，因此不应在全图中独立选择属性patch，而应先由候选
对象定位视觉区域，再在该区域及邻域中提取属性信息。

已有P8/P10只融合了composition CLS与composition patch，没有实现对象先定位、
属性后提取。此前3x3实验也只是平滑composition响应图，不是对象条件化邻域。

## 2. 固定结构

原Troika的composition、attribute、object三个CLS分支全部保留。对每个候选
组合`(attribute a, object o)`额外计算：

1. 使用ViT第12层patch和对象文本得到对象响应图。
2. 使用ViT第6层patch和属性文本得到属性响应图。
3. 用对象响应图对属性响应加权，得到`local_attr(a | o)`。
4. 以小残差加入原CLS组合分数：

`score(a,o) = score_cls(a,o) + alpha * local_attr(a | o)`

选择第12层定位对象，是因为此前它的翻转top-1一致率最高；选择第6层提取属性，
是因为它在固定checkpoint诊断中提供了最强局部互补信号。两个选择在运行前固定，
不重新扫描layer。

## 3. 区域构造

验证两个预注册版本：

- `soft`：对象patch相似度在空间维以temperature=0.07做softmax，作为连续权重。
- `hard3`：取对象最高响应patch，以它为中心构建3x3均匀邻域；边界处按实际
  patch数量重新归一化。

soft是主方案，hard3用于直接验证“对象中心周围一圈包含属性”的假设。不会扫描
其他kernel大小。

## 4. 融合与候选

CLS权重始终完整保留，局部属性仅作加性残差。只比较：

- 纯CLS；
- soft，alpha=0.05/0.10/0.20；
- hard3，alpha=0.05/0.10/0.20。

不使用alpha=1，不允许patch替换CLS，不修改attribute/object原始CLS logits。

## 5. 两阶段协议

### 阶段A：固定checkpoint验证

- D4：P0 seed0最佳CLS checkpoint。
- D5：P9 seed1最佳CLS checkpoint。
- 只运行UT-Zappos验证集。
- 每个seed都必须精确复现自己的CLS验证AUC。
- 同时测量真实对象响应在水平翻转下的top-1一致率、soft map相关性、3x3区域IoU
  和响应熵。
- 两组可在两张空闲GPU上并行，预计单组只需约4GB显存。

### 阶段B：结果门控

只有D4和D5都由验证AUC选择非CLS候选，且方向一致，才执行：

1. 固定统一配置；
2. 每个checkpoint只读取一次测试集；
3. 再决定是否实现端到端训练。

如果任一seed选择纯CLS或出现相反方向，则不读测试、不做完整训练，记录阴性结果。

## 6. 成功标准

主要标准：两个seed的验证AUC都超过各自CLS。辅助观察：Attr准确率、Seen/Unseen、
HM，以及对象区域翻转稳定性。测试结果不参与配置选择。

若soft优于hard3，说明连续对象区域比硬中心邻域可靠；若hard3更好，支持对象中心
周围局部属性假设；若两者都失败，说明CLIP对象响应不足以作为属性区域先验，或
当前属性文本/patch层不具备可用局部对齐，不继续投入完整训练。

## 7. 证据与输出

- 代码：`code_v6/object_conditioned_attr_sweep.py`
- 核心算子：`code_v6/model/object_conditioned.py`
- 单元测试：`code_v6/tests/test_object_conditioned.py`
- 队列：`scripts/stage6_queue.py`
- 状态：`logs/stage6_queue_state.json`
- 结果：`diagnostics/D4_objcond_attr_seed0_val/`、
  `diagnostics/D5_objcond_attr_seed1_val/`

完成后将结果追加到总报告，并同步原始JSON到本地镜像。

## 8. 固定 Checkpoint 结果与端到端确认

D4/D5均选择`soft:0.10`，验证AUC分别提升0.4651和0.3011个百分点，满足门控。
D6/D7固定配置测试一次，AUC分别变化+0.3959和-0.2333，均值基本持平。

因此端到端阶段固定为：

- P11：UT-Zappos seed0；
- P12：UT-Zappos seed1；
- object layer12、attribute layer6；
- soft temperature 0.07、alpha 0.10；
- 15轮，与P0/P9使用相同训练协议；
- 最多两张GPU并行，完整训练启动门槛37000 MiB。

P11只与P0比较，P12只与P9比较。训练期间不再更改结构或超参数。
