# CZSL Patch 作用验证总报告

更新时间：2026-09-17 10:25 CST  
远端项目：`./czsl/sparse_patch_troika`  
基线代码：Troika，原始目录只读：`./troika`

## 1. 研究问题

本轮实验集中回答以下问题：

1. 直接用 CLIP ViT patch token 替换 Troika 的 CLS 视觉评分，能否提高 CZSL？
2. 删除 CLS、增加参与聚合的 patch 数量，是否能改善组合识别？
3. patch 应该作用于全部三个分支，还是只作用于 composition 分支？
4. 最后一层 patch 是否包含可直接叠加到 CLS 上的互补信息？
5. 中间层 patch 是否比最后一层更适合属性-物体组合识别？
6. patch 的 hard top-k 选择是否稳定，邻域平滑是否有效？
7. 固定诊断得到的 layer6 residual 方案在端到端训练和不同 seed 上能否复现？
8. 对象条件化 soft patch 属性残差能否在双 seed 端到端训练中复现固定 checkpoint 信号？

当前实现是受控的 Troika patch pooling / residual 研究，不是完整 SAM 复现。
它保留 Troika 的 Adapter 和 CMT，没有加入 SAM 的 LoRA、VAC、KL、memory bank
等模块，因此不能直接用本实验结论否定或确认完整 SAM。

## 2. 实验协议与工程修正

数据集为 UT-Zappos，当前均为闭集 CZSL。训练使用 batch size 64、学习率
0.0005、15 epochs，并只依据验证集 AUC 选择 checkpoint。测试集不用于选择
layer、top-k、融合权重或其他超参数。

为保证 patch 变体和 CLS 基线公平，本项目修正了三个会干扰结论的问题：

- 原训练代码在验证后没有恢复 `model.train()`；新实验每轮明确恢复训练模式。
- 三个分支重复执行 patch LayerNorm，导致普通前向与缓存文本前向不一致；新代码
  统一为一次规范化。
- composition 分块评估时属性/物体词表被重复拼接；新评估器只编码一次 primitive
  文本，避免在更大组合空间中产生错误。

另外，服务器 NVML 用户态版本与内核驱动不一致。GPU 查询通过单进程
`LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.535.104.05` 解决，没有修改
系统驱动、共享环境或其他用户进程。

完整训练观测到约 32 GB CUDA allocation。调度器要求启动前至少 37000 MiB
空闲显存，项目并发硬上限为三张 GPU，不终止或抢占其他用户任务。

## 3. 代码版本

| 版本 | 主要内容 |
|---|---|
| `code/` | 初始 CLS、max、top-k patch pooling 对照 |
| `code_v2/` | 增加 `patch_branches`，允许只修改 composition 分支 |
| `code_v3/` | 固定 P0 checkpoint 的 CLS-patch 分数融合扫描 |
| `code_v4/` | ViT layer6/12/18/24 特征截取、稳定性和平滑诊断 |
| `code_v5/` | 可训练的 layer6 top-k2 composition residual 方案 |
| `code_v6/` | 对象条件化属性固定 checkpoint 诊断与稳定性分析 |
| `code_v7/` | 固定配置的对象条件化属性残差端到端训练 |
| `code_v8/` | 对象mask的uniform/center/wrong/shuffled机制控制实验 |

原始 Troika 和 CAMS 目录均未修改。每组训练在
`experiments/<run_id>/` 下保存配置、代码来源、训练进度、验证历史、最佳 checkpoint
和最终结果。

## 4. 已完成实验汇总

以下结果均为 UT-Zappos seed0。数值已经乘 100，单位为百分点。

| 实验 | 核心设置 | 最佳验证轮 | 验证 AUC | 测试 AUC | 测试 HM | 结论 |
|---|---|---:|---:|---:|---:|---|
| P0 | 新协议 CLS 基线 | 3 | **50.5867** | **42.5326** | **56.3985** | 公平主基线 |
| P1 | max，含 CLS，全部分支 | 1 | 48.4402 | 38.7292 | 52.2656 | 明显退化 |
| P2 | max，排除 CLS，全部分支 | 3 | 47.1255 | 41.0924 | 55.6500 | 未超过 CLS |
| P3 | top-k2，排除 CLS，全部分支 | 1 | 49.1946 | 38.8864 | 52.6292 | 验证优于 P2，但测试更差 |
| P4 | top-k4，排除 CLS，全部分支 | 1 | 47.6681 | 38.7594 | 51.8139 | 不支持继续增大 k |
| P5 | max，仅 composition 分支 | 2 | 49.3889 | 41.1836 | 54.4364 | 保留 primitive CLS 可恢复损失，但仍低于 P0 |

P0 测试的 Seen / Unseen / Attr / Obj 分别为 66.0802 / 73.7176 / 51.0295 /
72.4777。

### 4.1 直接 hard patch pooling

P1 和 P2 检验了 max pooling 中 CLS 是否参与候选集合。两者验证和测试 AUC
都没有超过 P0。P2 相比 P1 测试 AUC 较高，但验证 AUC 更低，说明不能根据测试集
排序选择方案。直接对全部分支使用 hard max 缺少正向证据。

### 4.2 top-k 数量

P3 将单 patch max 改为 top-k2 log-mean-exp，验证 AUC 相比 P2 提高 2.0691，
但测试 AUC 下降 2.2060。P4 继续检查 top-k4，验证 AUC 反而降到 47.6681。
因此没有继续扫描 k=8，避免在单 seed 上扩大超参数搜索。

### 4.3 patch 作用分支

P5 只在 composition 分支使用 patch max，属性和物体分支保留 Troika 原有 CLS
解耦向量。相对 P2，P5 验证 AUC 提高 2.2634、测试 AUC 提高 3.0912；但相对
P0 仍分别低 1.1978 和 1.3490。

这一结果说明：把 primitive 属性/物体分支也替换为 patch pooling 是前面性能下降
的重要来源之一。后续 patch 方法应保留属性/物体 CLS 分支，只修改组合分支。

## 5. 固定 checkpoint 的互补性诊断

### 5.1 P6：最后一层 patch 融合

P6 加载 P0 最佳 checkpoint，不重新训练，仅在 composition 分支计算：

`score = (1 - alpha) * score_cls + alpha * score_patch`

| 配置 | 验证 AUC |
|---|---:|
| CLS，alpha=0 | **50.5867** |
| layer24 max，alpha=0.10 | 50.4953 |
| layer24 top-k2，alpha=0.10 | 50.4762 |
| layer24 max，alpha=0.25 | 50.1335 |
| layer24 top-k2，alpha=0.25 | 50.0420 |
| layer24 max，alpha=1.00 | 12.8060 |
| layer24 top-k2，alpha=1.00 | 15.6428 |

验证集选择纯 CLS，测试复现 P0 的 42.5326。最后一层 raw patch 即使只占
10% 权重也会轻微降低验证 AUC，且权重增大后快速退化。因此停止继续扫描最后一层
的 k 和 alpha。

### 5.2 P7：中间层 patch

P7 在一次视觉前向中截取 ViT 第 6、12、18、24 层。最终 CLS、CMT 和 primitive
分支保持不变，只改变 composition patch 相似度使用的层。

| 配置 | 验证 AUC | 相对 P0 |
|---|---:|---:|
| CLS | 50.5867 | 0.0000 |
| layer6 max，alpha=0.10 | 50.9885 | +0.4017 |
| layer6 top-k2，alpha=0.10 | 51.0472 | +0.4605 |
| layer6 top-k2，alpha=0.25 | **51.2975** | **+0.7107** |
| layer12 top-k2，alpha=0.10 | 51.0413 | +0.4545 |
| layer18 max，alpha=0.10 | 50.7733 | +0.1866 |
| layer24 max，alpha=0.10 | 50.4953 | -0.0915 |

验证集选出的 layer6 top-k2 alpha=0.25 在测试集得到 42.5113，而 P0 为
42.5326，差 -0.0213，基本持平。该结果只说明中层 patch 可能包含末层没有的
局部互补信息，不能视为最终性能提升。

P7 的 layer24 alpha=0.1 精确复现 P6，对应实现正确性检查通过。

## 6. Patch 选择稳定性

固定 P0 checkpoint，在验证集 3214 张图上比较原图和水平翻转图。翻转后的
16x16 patch 坐标映射回原位置后，统计 hard 选择和完整响应图的一致性。

| 层 | top-1 一致率 | top-k2 Jaccard | 响应图相关 | argmax 距离 | margin | 归一化熵 |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 9.68% | 0.104 | 0.679 | 0.255 | 0.00849 | 0.9887 |
| 12 | **31.64%** | **0.319** | 0.762 | **0.150** | 0.00709 | 0.9867 |
| 18 | 25.98% | 0.253 | **0.787** | 0.235 | 0.00626 | 0.9842 |
| 24 | 16.30% | 0.143 | 0.597 | 0.239 | 0.00763 | 0.9798 |

所有层的 hard top-1/top-k2 都不稳定，且 top1-top2 margin 很小。layer12 的
离散选择最稳定，layer18 的完整响应图相关性最高。P7 验证最优的 layer6 反而
top-1 一致率最低、响应熵最高，因此 layer6 的小幅融合收益更可能来自弥散的局部
补充，而不是稳定的稀疏目标定位。

## 7. 固定邻域平滑

为了判断 hard patch 对轻微位移是否过于敏感，在 layer6 和 layer12 响应图上使用
replicate padding 的固定 3x3 平均，只比较验证集，不读取测试集。

| 层 | 平滑后最佳配置 | 平滑验证 AUC | 未平滑同层最佳 |
|---:|---|---:|---:|
| 6 | top-k2，alpha=0.10 | 51.0711 | **51.2975** |
| 12 | top-k2，alpha=0.10 | 50.7976 | **51.0413** |

平滑后仍略高于纯 CLS，但低于未平滑候选。固定邻域平均没有成为新的改进点，
因此停止继续扫描 kernel 大小。

## 8. 正确性与资源验证

在进入完整训练前执行了以下检查：

- pooling 单元测试覆盖 CLS、max、排除 CLS、top-k 和 composition-only 行为。
- `code_v5` 的 7 个 CPU 测试全部通过。
- 真实 batch=2 的普通前向与缓存文本前向最大误差为 0。
- Adapter 和保留的属性/物体分支均获得有限、非零梯度。
- P7 第 24 层结果精确复现 P6，排除多层截取实现偏差。
- 已完成完整训练均保存 provenance、配置、验证历史和峰值显存。

完整训练峰值约 31.94-32.25 GB；固定 checkpoint 诊断约 2.8-3.9 GB。

## 9. 端到端验证状态

截至 2026-09-16 21:03 CST，P8、P9、P10 均已完成。

| 实验 | seed | 状态 | 最佳验证 AUC | 最佳轮次 | 测试 AUC |
|---|---:|---|---:|---:|---|
| P8 layer6 residual | 0 | 已完成 | 49.5535 | 3 | 43.3609 |
| P9 CLS | 1 | 已完成 | **51.4583** | 5 | 39.1700 |
| P10 layer6 residual | 1 | 已完成 | 50.2769 | 3 | **44.5784** |

P8 和 P10 使用相同固定方案：layer6、top-k2、只修改 composition 分支、
residual alpha=0.25；P8 为 seed0，P10 为 seed1。P9 是与 P10 配对的 seed1
CLS 基线。

两个seed的正式配对结果：

- P8 比 seed0 CLS 验证 AUC 低 1.0333，按预定规则为阴性确认。
- P8 测试 AUC 高 0.8283、HM 高 1.0777，但不能根据测试集反转验证结论。
- P10 比 seed1 CLS 验证 AUC 低 1.1814，但测试 AUC 高 5.4084。
- 两seed residual验证AUC平均低1.1073，测试AUC平均高3.1183。
- 验证与测试排序在两个seed上稳定相反，需作为分布/校准问题分析，不能把测试
  提升直接声明为方法涨点。

当前所有训练均已结束，相关GPU已释放。

## 10. 当前结论

1. 直接删除 patch 或使用 patch 并不天然提高 CZSL；作用位置和分支比是否包含
   CLS 更关键。
2. 末层 hard max/top-k 缺少互补证据，尤其不适合直接替换全部三个分支。
3. 属性/物体 primitive 分支应保留 Troika 原有 CLS 解耦表示。
4. 中间层 patch 比最后一层更有希望；layer6 在固定 checkpoint 验证集出现
   +0.7107 的信号，但测试持平。
5. hard patch 索引对水平翻转非常不稳定，当前证据不支持把方法解释成稳定的稀疏
   区域定位。
6. 固定 3x3 平滑不能提高未平滑候选，不值得继续做手工 kernel 搜索。
7. 两个seed均确认端到端layer6 residual验证AUC低于CLS，因此关闭该hard top-k
   residual路线；但重复出现的测试AUC上升值得做独立分布诊断。

固定checkpoint的诊断收益没有在两个seed的端到端验证结果中复现。

## 11. 后续决策标准

1. 不启动seed2，不再扫描layer、k、alpha或固定kernel。
2. 先比较P0/P8、P9/P10完整bias曲线及验证/测试类别子集，解释排序反转。
3. 下一条方法诊断是对象条件化soft patch aggregation；若响应仍不稳定，再加入
   原图/翻转图响应一致性约束。
4. 任意新方案先做固定checkpoint和低显存诊断，再决定是否投入完整训练。

## 12. 证据位置

- 首轮完整结果：`results/STAGE1_20260916.md`
- top-k 与分支消融：`results/STAGE2_20260916.md`
- 固定 checkpoint 与多层诊断：`results/STAGE3_20260916.md`
- P8 端到端 seed0 确认：`results/STAGE4_20260916.md`
- seed1 配对和两seed汇总：`results/STAGE5_20260916.md`
- 稳定性和平滑：`results/STABILITY_20260916.md`
- 决策记录：`docs/DECISIONS.md`
- 过程记录：`docs/PROGRESS_20260916.md`
- 训练日志：`logs/P0_*.log` 至 `logs/P10_*.log`
- 原始配置和指标：`experiments/<run_id>/{config,provenance,history,result}.json`
- 诊断原始 JSON：`results/evidence/`

当前hard top-k residual验证阶段已结束，下一阶段应先解释验证/测试排序反转。

## 13. 对象条件化属性残差最终确认

第六轮在固定 P0/P9 CLS checkpoint 上得到一致但较小的验证增益：seed0为+0.4651，
seed1为+0.3011。随后严格固定 object layer12、attribute layer6、soft temperature
0.07和alpha0.10，运行P11/P12双seed端到端确认。

| seed | CLS验证AUC | 方法验证AUC | 验证差值 | CLS测试AUC | 方法测试AUC | 测试差值 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | **50.5867** | 49.0116 | -1.5752 | **42.5326** | 32.1410 | -10.3917 |
| 1 | **51.4583** | 49.9049 | -1.5534 | **39.1700** | 38.0155 | -1.1545 |
| 均值 | **51.0225** | 49.4582 | **-1.5643** | **40.8513** | 35.0783 | **-5.7731** |

两个seed的验证和测试均退化。测试属性准确率平均下降2.0247点，物体准确率仅上升
0.3088点，说明当前残差没有实现“用对象区域patch补充细粒度属性”的目标。固定
checkpoint的推理期互补信号在共同训练中消失，结合soft map熵约0.989，当前响应更
接近弥散的对象条件化重加权，而不是可靠定位。

最终决策是关闭当前端到端对象条件化残差路线：不运行seed2，不继续扫描layer、
kernel、temperature或alpha。若继续patch研究，应首先验证冻结/stop-gradient对象
mask、增强一致性或校准门控能否阻止训练期漂移，而不是继续扩大patch数量搜索。

完整记录见`results/STAGE6_20260916.md`和`results/STAGE7_20260916.md`；P11/P12
原始配置、历史、来源和结果已同步到`results/evidence/`。

## 14. 对象 Mask 机制控制结论

端到端对象条件化残差失败后，D8/D9在P0/P9固定checkpoint上比较uniform、center、
correct-object、wrong-object和跨图shuffled mask。正确对象mask在两个seed上均比
uniform/center提供更多纯局部属性信息，但没有表现出可靠的对象语义特异性。

- seed0 correct-object局部属性准确率10.2054，wrong-object为10.1742，差距仅0.0311；
  pair AUC还低于对象无关的center先验。
- seed1 correct-object局部属性准确率24.3622，shuffled mask为24.4866；其pair AUC
  虽是patch控制组最高，仍低于CLS 0.2466点。
- correct/wrong/shuffled的mask熵均约0.988，接近均匀。

两个seed均未通过预注册门控，故不训练基于当前mask的条件属性head。综合P0-P12与
D0-D9结果，结论是“patch包含属性信息，但原始CLIP对象相似度不足以作为可靠定位器”。
下一阶段转向CAMS latent cross-attention低成本诊断，让latent从全部patch聚合证据，
而不是继续搜索hard patch数量或soft mask超参数。完整记录见
`results/STAGE8_20260917.md`。

## 15. CAMS Latent 对象条件化诊断

Stage 9使用CAMS的32个Gated Cross-Attention latent units，尝试用对象分支unit与
对象prompt相似度选择属性unit。固定两个原复现checkpoint，只评估与CAMS训练一致的
closed-world validation。uniform重建原mean pooling，最大pair score误差3.81e-6。

| seed | uniform AUC | correct AUC | wrong AUC | shuffled AUC | correct局部属性 | correct熵 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 45.171352 | 45.171328 | 45.171352 | 45.171352 | 30.6161 | 0.999122 |
| 1 | 48.985751 | 48.985751 | 48.985751 | 48.985751 | 33.5408 | 0.999407 |

对象权重几乎完全均匀，correct-object在两个seed上都没有超过uniform、wrong-object
和跨图shuffled控制。seed1中wrong/shuffled的局部属性准确率还比correct高0.0311点。
因此两个预注册门控均失败，不进入可训练对象条件latent head，也不扫描temperature。

当前证据把问题进一步缩小为：视觉局部信息存在，但现成的CLIP patch对象相似度和
CAMS对象latent相似度都不能可靠地决定“哪个局部证据服务于哪个对象”。下一步只能
先检查attribute latent units是否本身存在可利用的单元分工上界，再决定是否设计带
多样性和增强一致性约束的属性专用latent。完整记录见`results/STAGE9_20260917.md`。

## 16. Attribute Latent 分工上界

Stage 10不再尝试对象选择，而是直接检查32个CAMS attribute units是否存在可利用的
属性专家分工。比较原mean、独立unit logits平均、属性prompt max、log-mean-exp和
全验证集随机跨图控制，并计算oracle上界。

| seed | mean AUC | max AUC | LME AUC | mean属性 | 任一unit属性 | 不同预测数 | unit余弦 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 45.1714 | 45.2937 | 45.2517 | 30.6161 | 33.0429 | 1.1481 | 0.9743 |
| 1 | 48.9858 | 48.7184 | 48.9847 | 33.5408 | 35.2831 | 1.0868 | 0.9836 |

seed0的prompt max有+0.1224点小幅AUC信号，但seed1下降0.2673点。oracle任一unit
正确率只比mean高2.43/1.74点，远低于预注册5点门槛；按真实属性target选择unit也只
提高0.28/0.34点。32个units在每张图上几乎给出同一个属性判断，且特征余弦接近1，
没有形成可供router利用的专家分工。

至此，当前视觉路线的两种选择器都被排除：raw CLIP patch对象mask缺少对象特异性，
CAMS现有latent units又高度冗余。停止在现有表示上继续搜索unit、temperature和融合
权重。若以后重新做视觉端，需要在训练目标中显式创建属性query分工与一致性；近期更
值得优先处理分数校准及验证/测试排序反转。完整记录见`results/STAGE10_20260917.md`。

## 17. OCAQ 监督属性 Query 验证

Stage 11建立独立`ocaq_troika/`，完整保留Troika CLS分支，新增第12层对象Query和
第6层属性Query。四个属性Query分别从核心、核心/邻域混合、偏邻域、近全局空间角色
开始；组合与属性残差均使用零初始化gate。对象、属性、错误对象特异性、注意力多样性
和负载均衡均有显式监督。

工程验证中，零gate与P0输出最大误差为0，冻结Troika无梯度。seed0一轮训练后，
Query注意力重合度从约0.994降到0.049，有效Query数3.999，证明监督可以创建现有
CAMS latent所缺少的分工。

固定结构在P0/P9上分别训练五轮，只用validation选模：

| seed | 基线AUC | OCAQ AUC | AUC变化 | 属性准确率变化 | 最佳轮 |
|---:|---:|---:|---:|---:|---:|
| 0 | 50.5867 | 51.3522 | +0.7655 | -0.2178 | 4 |
| 1 | 51.4583 | 51.6175 | +0.1592 | -0.3734 | 2 |

固定checkpoint gate消融表明，仅组合残差在两个seed上分别提高AUC 0.6136和0.1902
个百分点，是可重复的正信号；仅属性残差对属性准确率只提高0.0933和0.0622个百分点，
无法成为强属性专家，且seed1 AUC下降。完整OCAQ平均AUC提高0.4623个百分点，但两个
seed属性准确率均下降。

因此当前版本未通过预注册门槛：不解冻Troika、不进入Q3/Q4、不读取test。结论是
“监督Query能够形成有效视觉分工并改善部分组合排序，但当前属性判别和CLS/局部校准
不足，尚未解决CZSL的属性识别问题”。后续应先加强属性专家训练和校准，不继续扫描
Query数、patch数、层数或邻域大小。完整中文记录见
`docs/OCAQ_TROIKA_PLAN_20260917.md`和`results/OCAQ_STAGE_20260917.md`。

## 18. TGAQ 属性文本 Query 与机制消融

TGAQ把每个属性文本直接作为Query，在第12层对象区域及其3x3邻域内，从第6层patch
提取属性视觉证据。冻结Troika，只训练TGAQ；损失由标准/逆平方根类别平衡属性CE、
对象Query CE和正确/错误对象特异性margin组成。

双种子正式训练均失败：seed0/1最佳局部属性micro为32.4207%/31.4250%，分别低于
Troika原属性头2.3958/1.8357点；macro下降1.6500/6.9690点。对象Query准确率达到
75.67%/78.06%，但正确对象相对最强错误对象的属性margin为-0.0935/-0.2063，说明
对象识别正确并没有建立对象区域与属性证据的正确联系。

固定checkpoint机制消融进一步显示：错误对象区域或跨图区域与正确区域持平或略高，
当前mask不具对象语义特异性。仅保留属性文本Query和区域时，micro达到39.3279%和
34.5675%，超过Troika；但macro仅11.9253%和10.6405%，预测集中到少数高频属性。
完全去掉区域后micro又降至19.4462%/23.7399%。第6层原始patch-text mean/max仅
10.45%-17.49%，明显弱于学习后的Query。

因此视觉端仍有局部属性信息，但下一步不能再围绕对象mask、patch数量或邻域大小搜索。
合理方向是先验证类别先验校准和CLS稳定融合，确认高频塌缩是否可校正；若双种子CZSL
validation AUC和属性macro不能同时受益，则应重做无对象mask的多层属性表征训练。
完整结果见`results/ATTRIBUTE_EXPERT_V2_20260917.md`。

## 19. TGAQ 校准与 CLS 小残差

固定T1/T2后，对属性-only局部分数使用训练类别先验校准，再以小残差加入CLS。seed0
最优为不校准、alpha0.05，AUC从50.5867升到51.0895；seed1最优为tau1、alpha0.05，
AUC从51.4583升到51.5721。由于两个seed的最优配置不一致，严格门槛失败，不读test。

共同的不校准pair-only/alpha0.05在两个seed仍分别提高0.5028和0.1000点，重复确认
patch Query能补充少量组合排序信息；但把它加入属性分支没有稳定收益。训练先验校准
也没有恢复macro，反而使局部专家的micro/macro进一步下降。这表明问题不只是分数
偏置，而是局部属性表示缺少稳定的长尾类结构。

因此停止当前TGAQ和后验校准，不再搜索tau/alpha。下一版应去除对象mask，使用多层
属性Query直接建立语义和细节对应，并以稀疏注意力替代手工patch数量选择；CLS继续
承担全局组合和primitive稳定性。
