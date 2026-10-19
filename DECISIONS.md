# Decision Log

## 2026-09-15: baseline selection

Selected Troika rather than CAMS for the first validation stage.

Reasons:

- The SAM paper reports a direct Troika + Sparse Alignment ablation.
- Troika already exposes CLS and patch tokens in its visual encoder.
- CAMS contains gated latent-query branches that would confound the first
  patch-selection experiments.
- A complete five-seed Troika reproduction is available for comparison.

Successful modules can later be transferred to CAMS as a stronger baseline.

## 2026-09-15: pooling implementation

Sparse pooling is parameter-free and isolated in `model/patch_pooling.py`.
The original CLS path remains the default to preserve checkpoint behavior.
Similarity pooling happens in normalized cosine space before CLIP's learned
logit scale is applied.

## 2026-09-16: controlled protocol v1

The original train.py enters model.eval() during validation and fails to call
model.train() in subsequent epochs. New experiment.py restores training mode
at every epoch for every arm, including a fresh CLS control. Historical results
are reference-only, not the primary improvement denominator.

The original forward normalizes the same patch sequence repeatedly inside the
three-branch loop; forward_for_open normalizes it once. normalize_patch_once=True
aligns these paths for all fresh runs; the legacy default remains False.
Sparse logits use the raw projected CLIP output tokens followed by L2
normalization. CMT keeps its own LayerNorm. CLS inclusion only controls the
pooling candidate set: CLS still participates in the visual encoder and CMT.

The new evaluator encodes attribute/object texts only once across composition
chunks. The upstream helper concatenates duplicated primitive vocabularies,
which becomes incorrect when the number of compositions exceeds a text batch.
This matters for MIT-States and C-GQA.

The new method retains Troika's adapters and CMT while replacing the final
visual scoring operation on all three branches. It is a controlled Troika
pooling variant, not a full SAM reproduction (no SAM LoRA/VAC/KL/memory bank).
It cannot be claimed to reproduce SAM Table 7 without official integration code.

NVML user-space 545 differs from loaded kernel 535. A per-process LD_PRELOAD
of the existing matching 535 NVML library restores GPU queries without editing
system libraries, drivers, or the shared conda environment.

Only a finite three-run UT pilot is queued, with at most two simultaneous GPUs.
The scheduler prefers empty GPUs, then largest free memory (minimum 28,000 MiB).
It stops launching new jobs after failure or logs/STOP_QUEUE. No other process
is terminated. Hyperparameters remain fixed across the first three arms.

## 2026-09-16: stop broad top-k search after P4/P5

P4 top-k=4 did not preserve the validation improvement observed at top-k=2.
P5 recovered most of the loss from all-branch patch scoring by retaining the
attribute/object CLS disentanglers, but remained below the fresh CLS control.
Therefore k=8 and broad seed/dataset expansion are deferred. Primitive CLS
branches remain mandatory in subsequent diagnostics.

## 2026-09-16: fixed-checkpoint residual fusion before new training

P6 tests whether final-layer patch scores already contain complementary signal
inside the best P0 representation. Candidate fusion weights are selected only
on validation, and test is evaluated once for the selected candidate. Pure CLS
won validation; even alpha=0.1 reduced AUC for max and top-k2. Direct score
fusion and further alpha tuning are stopped.

The next smallest diagnostic is E4 layer location. P7 keeps final CLS, final
CMT context, and primitive branches fixed, changing only the patch layer used
by the composition score. It captures layers 6/12/18/24 in one image forward.
This is a representation diagnostic, not evidence that a trained multiscale
selector will improve CZSL.

## 2026-09-16: one end-to-end layer-6 confirmation

P7 selected layer-6 top-k2 with alpha=0.25 on validation, improving validation
AUC by 0.7107 percentage points over the fixed P0 checkpoint. Its test AUC was
effectively tied with P0, so this is not a claimed test improvement. The signal
is sufficient for exactly one controlled training run, P8, with that selection
frozen before training. P8 is compared to P0 using validation AUC first.

No additional alpha, layer, k, seed, dataset, or binding variants are queued.
If P8 fails to beat P0 validation, the next action is spatial/selection
diagnostics rather than more hyperparameter search. If it does beat P0, repeat
the fixed P0/P8 comparison on additional seeds before adding architecture.

## 2026-09-16: use up to three GPUs with paired controls

The user confirmed that up to three GPUs may be used concurrently. Full
batch-64 training still requires at least 37,000 MiB free because observed peak
allocation is about 32 GiB and shared cards already hold unrelated processes.
Low-memory diagnostics may share cards with sufficient remaining memory.

While P8 seed0 runs, P9 CLS seed1 and P10 fixed-P8 seed1 form the next paired
comparison. The stage5 queue may run both when memory permits; together with
P8 the project hard limit remains three GPUs. A method seed is never compared
against a CLS checkpoint from a different seed.

Flip diagnostics show that hard patch selections are unstable at every tested
layer. Layer 12 has the best discrete consistency, while layer 6 has the best
validation fusion but the worst top-1 consistency. Fixed 3x3 smoothing reduces
the validation result relative to unsmoothed layer-6/layer-12 candidates, so
kernel scanning is stopped.

## 2026-09-16: P8 seed0 does not confirm the validation gain

P8 completed successfully. Its best validation AUC is 49.5535 versus 50.5867
for the paired P0 seed0 CLS control, a decrease of 1.0333 percentage points.
Its test AUC is 43.3609 versus 42.5326 for P0, but the protocol selects methods
on validation, so the higher test number is descriptive and does not reverse
the negative confirmation result.

P9/P10 continue as the seed1 paired comparison. Seed2 is not launched while
that pair is incomplete. If P10 also fails to exceed its same-seed CLS control,
the layer/k/alpha/kernel search is closed and the next work must target the
observed instability through object-conditioned soft aggregation or a minimal
transformation-consistency objective.

## 2026-09-16: close hard top-k residual search after seed1 pair

P9/P10 completed successfully. P10 validation AUC is 50.2769 versus 51.4583
for P9, a decrease of 1.1814 percentage points. Across seeds 0 and 1, the
residual method is lower on validation by 1.1073 points on average. Seed2 and
further layer/k/alpha/kernel scans are therefore not justified.

The test ordering is consistently opposite: residual test AUC is higher by
0.8283 and 5.4084 points for seeds 0 and 1, respectively. This is recorded as
a validation/test distribution or calibration diagnostic, not a method gain.
The next stage should first analyze full bias curves and per-composition
subgroups, then use a fixed-checkpoint object-conditioned soft aggregation
diagnostic if the discrepancy has a coherent source.

## 2026-09-16: object-conditioned attribute residual before new training

The next diagnostic keeps all original CLS branches and adds local evidence
only as a residual to the composition score. Layer 12 object-to-patch responses
define the object region because they had the best flip consistency; layer 6
attribute-to-patch responses provide the detail signal because layer 6 had the
strongest frozen-checkpoint complementarity.

Two pre-registered masks are compared: a soft spatial object distribution and
a literal 3x3 neighborhood around the object's argmax. Residual weights are
limited to 0.05, 0.10, and 0.20. P0 seed0 and P9 seed1 checkpoints are evaluated
on validation in parallel. Test evaluation and end-to-end training are gated on
both seeds selecting a non-CLS candidate. No ground-truth object is used in pair
scoring; the candidate pair's object text defines its region.

## 2026-09-16: allow one end-to-end object-conditioned confirmation

Both frozen CLS checkpoints selected the same soft-mask alpha=0.10 candidate.
Validation AUC improved by 0.4651 points for seed0 and 0.3011 for seed1. The
literal hard 3x3 neighborhood was weaker, consistent with its low flip IoU.

The one-time test evaluation was mixed (+0.3959 seed0, -0.2333 seed1), so it
does not establish a test gain. The repeated validation signal is sufficient
for exactly one fixed end-to-end pair using seeds 0 and 1. Object layer 12,
attribute layer 6, soft temperature 0.07, and residual alpha 0.10 are frozen.
No additional layer, kernel, temperature, or alpha search is allowed.

## 2026-09-17: close the current object-conditioned residual route

P11 and P12 completed successfully under the frozen Stage 7 configuration.
P11 seed0 validation AUC is 49.0116 versus 50.5867 for P0, a decrease of
1.5752 points. P12 seed1 validation AUC is 49.9049 versus 51.4583 for P9, a
decrease of 1.5534 points. The paired mean validation decrease is 1.5643.

Test AUC also decreases for both seeds: by 10.3917 points for seed0 and 1.1545
points for seed1. Mean test attribute accuracy falls by 2.0247 points, while
object accuracy rises by only 0.3088. The current residual therefore fails its
intended attribute objective and is not retained as a training method.

The frozen-checkpoint gain does not justify seed2 or further layer, neighborhood,
temperature, or alpha search. Any later patch method must first address training
stability and the diffuse object map, for example through a frozen/stop-gradient
mask, transformation consistency, or a tightly controlled calibration gate.
These are future hypotheses and are not launched in this stage.

## 2026-09-17: reject the CLIP object-similarity mask mechanism

D8/D9 compared the correct-object soft mask with uniform, center, wrong-object,
and cross-image shuffled controls on the paired P0/P9 validation checkpoints.
The correct mask contains some local attribute information relative to uniform
and center weighting, but it is not reliably object-specific.

For seed0, correct-object local attribute accuracy is 10.2054 versus 10.1742
for the wrong-object mask, while its pair AUC is lower than the center prior.
For seed1, correct-object has the best pair AUC among patch controls but remains
below CLS, and its local attribute accuracy is lower than a shuffled-image mask.
Both pre-registered gates therefore fail.

Do not train the frozen-mask conditional attribute head and do not scan the
current mask's layer, top-k, neighborhood, temperature, or residual weight.
The next permitted direction is a low-cost CAMS latent cross-attention diagnostic
that aggregates all patches and conditions attribute extraction on object
semantics without treating raw CLIP patch similarity as a localization mask.

## 2026-09-17: reject direct object-similarity weighting of CAMS units

C2/C3 reproduced the original CAMS closed-world validation checkpoints within
`3.81e-6` maximum pair-score error. Their baseline AUC values, 45.1714 and
48.9858, also match the original training logs, so the diagnostic uses the
correct model-selection protocol.

Object-prompt weighting of the 32 object latent units has normalized entropy
0.9991/0.9994. Correct-object weighting does not improve pair AUC or local
attribute accuracy over uniform, wrong-object, or shuffled controls in either
seed. Do not train this conditional head and do not sharpen it by scanning
temperature; the selector has no demonstrated semantic specificity.

The next allowable low-cost test is an attribute-latent specialization upper
bound: mean versus individual units, attribute-selected units, and an oracle
unit. A trainable specialized-latent method is justified only if this diagnostic
shows repeatable headroom in both seeds.

## 2026-09-17: close selection over existing CAMS attribute units

The attribute-unit specialization upper bound also fails. Across seeds 0 and 1,
an oracle that counts a sample correct when any of 32 units predicts the target
improves over mean pooling by only 2.43 and 1.74 percentage points. Selecting
the unit with the highest ground-truth target logit improves accuracy by only
0.28 and 0.34 points.

The units are nearly duplicates: they produce only 1.148/1.087 distinct
attribute predictions per image on average and have mean off-diagonal feature
cosine 0.974/0.984. Prompt max/LME has a small positive AUC change only in
seed0 and is neutral or negative in seed1.

Do not train a router over the existing units and do not scan unit count,
selection temperature, or fusion weights. A future visual route must explicitly
create specialization during training through supervised attribute queries,
diversity, and transformation consistency. This is a new architecture, not a
continuation of the current patch selector. The nearer-term research priority
is score calibration and the repeated validation/test ordering discrepancy.

## 2026-09-17: allow supervised OCAQ dual-seed validation

The new OCAQ branch is not a selector over the failed fixed patch/CAMS units. It
creates four new attribute queries with explicit object-query, attribute,
specificity, diversity, and load-balancing supervision while preserving the
Troika CLS paths through zero-initialized residual gates.

The engineering gate passes: zero gates reproduce P0 exactly, frozen baseline
parameters receive no gradients, and one seed0 epoch reduces attention overlap
from approximately 0.994 to 0.049 while retaining 3.999 effective queries. The
one-epoch AUC is neutral/slightly negative and is not used as evidence of a
gain. Permit one fixed five-epoch validation-only run for each of seeds 0 and 1.
Do not inspect test unless both pre-registered validation gates pass.

## 2026-09-17: keep the supervised-query hypothesis, reject the current fusion

Q1/Q2 show that supervised queries do specialize: both checkpoints retain about
four effective queries and have very low attention overlap. The full residual
improves validation AUC by 0.7655 points for seed0 and 0.1592 for seed1, but
attribute accuracy decreases in both seeds. The pre-registered gate therefore
fails and Q3/Q4 plus test evaluation are forbidden.

Fixed-checkpoint gate ablations locate the repeatable signal in the composition
residual (+0.6136/+0.1902 AUC points). The attribute residual improves attribute
accuracy by only 0.0933/0.0622 points and is not a sufficiently strong attribute
expert. Do not scan query count, layers, neighborhood size, gate weights, or loss
weights on this version. A future version must first improve supervised local
attribute discrimination and score calibration while retaining the CLS branch.

## 2026-09-17: reject object-semantic localization in TGAQ, retain local evidence

T1/T2 fail the standalone attribute gate, but D17/D18 identify two separate
effects. Replacing the learned correct-object region with a wrong-object or
cross-image region does not reduce validation attribute accuracy. The current
region prior therefore cannot be interpreted as object-semantic localization.

Removing object context while retaining the region raises micro attribute
accuracy above Troika in both seeds, but macro accuracy collapses by 15.35 and
19.76 points because predictions concentrate on frequent attributes. Removing
the region as well loses most of that micro signal. Raw layer-6 patch/text mean
or max is much weaker than the learned query output.

Do not continue tuning object masks or neighborhood geometry. The retained
hypothesis is narrower: learned local attention extracts useful attribute
evidence, but it needs class-prior calibration and CLS-level stabilization.
Permit one fixed-checkpoint, validation-only calibration/fusion diagnostic.
Require consistent dual-seed CZSL AUC improvement without further macro
attribute degradation before training another architecture or reading test.

## 2026-09-17: close post-hoc TGAQ calibration, retain pair residual evidence

D19/D20 reproduce the two CLS baselines exactly. Their individually selected
fusion candidates differ in prior-correction strength, so the pre-registered
same-configuration gate fails and test evaluation remains forbidden. A common
uncalibrated pair-only residual at alpha 0.05 nevertheless improves validation
AUC by 0.5028/0.1000 points, confirming a small composition-ranking signal.

Post-hoc class-prior correction does not repair the standalone local expert;
larger correction sharply reduces both micro and macro attribute accuracy.
Adding the local logits to the primitive attribute branch is also inconsistent.
Do not train a calibration gate or continue tuning tau/alpha.

The next architecture may retain a zero-initialized composition residual, but
must replace object-mask conditioning with an attribute-first multi-layer
representation. Use layer 12 for semantic keys, layer 6 for detail values, and
a sparse attention operator without a manually selected patch count. Preserve
all CLS branches and keep validation-only dual-seed gates.

## 2026-09-17: retain MSAQ attribute-first representation, reject attribute-only pair residual

MSAQ V3 proves that attribute-first sparse attention can learn useful local
visual structure without object masks: sparsemax retains 37%-54% zero weights,
and local macro attribute accuracy reaches 28.52%/31.56%. M1 improves CZSL AUC
by 0.7096 points, but M2 improves by only 0.0704; dual-seed AUC gate fails.

Fixed gate ablation shows the repeatable component is pair-only residual
(+0.6223/+0.0573), whereas primitive attribute residual is inconsistent and
harms seed1 AUC. Inspection identifies the reason: the V3 pair residual gathers
only an attribute-specific logit. All candidates sharing an attribute receive
the same local residual regardless of object, so the scorer cannot model an
attribute-object interaction.

Do not tune V3 gates or sparse parameters and do not read test. Allow a new
MPAQ architecture that keeps attribute-only spatial attention but scores each
attribute visual feature against its candidate composition text. This introduces
object specificity at the score level, not through an object localization mask.

## 2026-09-17：将局部组合证据限制为属性内对象校正

MPAQ V4的raw local pair scorer不能承担跨属性排序，但D23显示其在真实属性已知时能稳定
区分对象。V5因此不再让该分数改变属性级竞争：每个属性候选组内做零均值中心化，仅将对象
相对排序残差加入CLS composition logits；局部pair监督也改为同属性候选内CE。P1c一轮
seed0 AUC+0.2548点、macro不变，D24条件对象Top-1为66.7704%，通过机制门槛。

固定V5配置，允许P1/P2双seed五轮validation-only验证。禁止测试集读取、层位/损失/门控
扫描；只有双seed共同通过AUC和macro门槛才继续。

## 2026-09-17：关闭 MPAQC V5 的训练路线，保留条件对象诊断

固定五轮结果为P1/seed0 AUC+0.2780点、P2/seed1 AUC+0.0089点，两个seed的primitive
macro均不变。P2未达到+0.2点AUC门槛，故不读取test，不再修改该版的残差形式、损失权重、
层位或gate。局部属性条件对象分数仍可能真实有效，但它没有稳定改善CLS最难的组合错误。

允许且只允许对P1/P2最佳checkpoint做零训练、validation-only互补性诊断：在真实属性
已知的候选对象集合中比较CLS、局部与融合，统计错误重叠和oracle上界。结果决定下一版是
做不确定性条件融合，还是放弃局部对象残差路线。

## 2026-09-17：局部对象信号有有限互补性，但不能全局启用

D25/D26中CLS条件对象Top-1显著高于局部scorer，但局部仍正确处理约3.4%的CLS错误，oracle
上界较CLS高约3.4点。与此同时局部会破坏14.8%/19.8%的CLS正确样本，故单一全局gate在
统计上不可能稳定有效。允许下一步做无训练的CLS置信度五分位诊断；只有互补正确样本明显
富集于最低置信分位，才允许设计样本级不确定性gate。

## 2026-09-17：关闭基于 CLS 置信度的局部对象融合

D27/D28最低置信分位中，局部纠正的CLS错误少于其破坏的CLS正确样本，两个seed方向一致。
因此CLS top-1/top-2间隔不足以选择局部对象证据；禁止训练不确定性gate、扩大局部residual
或继续当前MPAQC对象条件分支。此前否定的是CLIP文本attention生成的对象区域，不等同于
真实前景或检测器box。允许下一阶段仅先验证独立前景定位的可行性和语义质量。

## 2026-09-17：确认MSAQ已使用真实前景，停止空间约束搜索

F0b确认UT-Zappos白背景最大连通域是可信主体前景；D29/D30进一步表明M1/M2属性attention
显著富集于该前景，最大注意力patch几乎总在前景核心。故用户的对象优先直觉并未被否定，
但当前属性Query已实现它。禁止训练foreground硬mask、core/ring、中心邻域或再次使用文本
对象mask；下一版必须针对前景patch表征质量或融合兼容性，而不是空间定位。

## 2026-09-17：拒绝带真实属性身份的密集Query监督

DAPQ seed0一轮的true-query dense macro达到95.22%，但把图像替换为固定偏移的其他鞋图后
仍有79.00%，且两种图像下Query特征余弦为0.9572。该监督允许第y个属性文本Query直接携带
目标y，形成标签捷径，不能作为patch属性证据。禁止继续DAPQ双seed、损失权重扫描或test。

下一版必须移除真实属性条件Query。允许ICAQ使用冻结图像attribute CLS产生唯一Query，在
layer12语义patch上定位并从layer6读取细节，对全部属性文本原型分类。必须通过图像错配
审计后才可进入双seed正式训练。

## 2026-09-17：保留ICAQ图像属性证据，拒绝冻结视觉端直接融合

ICAQ双seed错配审计均通过，说明单图像Query确实从patch读取属性信息；但固定五轮AUC为
+0.1690/-0.1244点，未达到双seed门槛。禁止继续扫描composition gate、patch层位或稀疏
参数，不读取test。

允许下一阶段只解冻ViT最后两个block的既有attention/FFN Adapter，并用冻结Troika教师对
学生加局部residual前的三个CLS logits做在线KL蒸馏。该阶段只解决局部属性空间与CLS空间
不兼容的问题；若仍失败，不再扩大视觉端解冻范围。

## 2026-09-17：关闭扩大视觉端解冻范围的路线

ICAQA C0c中在线教师将primitive macro变化控制在-0.1468点，局部属性macro达到31.3210%，
但AUC仍下降0.2336点，未过一轮门槛。故不启动双seed，不读取test，不调整蒸馏权重、
Adapter学习率或解冻更多层。

现有证据已经把问题定位为“真实局部属性证据无法通过属性级加性residual稳定改善组合排序”，
而不是patch未找到主体、局部属性不可识别或视觉容量不足。后续若继续，必须改为显式属性-对象
关系或组合级训练目标；禁止继续patch数量、中心邻域、全局gate和视觉解冻范围搜索。

## 2026-09-18：属性是主要错误来源，但固定ICAQ不足以启动RCR

D37/D38显示Troika组合错误中75%-80%属于对象正确、属性错误，错误样本的真实组合
Top-10覆盖超过91%，因此主要问题不是对象识别或候选召回。ICAQ真实属性Top-5覆盖
74%-79%，说明patch中存在软属性信息；但其Top-1只有约31%，与Troika关键错误的核心
可修正交集为2.0535%/1.5868%。seed1未达到预注册2%门槛，双seed失败。

禁止使用当前固定ICAQ Top-1专家训练RCR，禁止扫描融合gate、候选数或权重，也不读取
test。若继续视觉端，必须作为新架构重新训练候选对称、组合感知的属性Query：所有候选
统一计算、不得按真实标签选择Query，直接使用同对象错属性与同属性错对象负样本，并保护
Troika高置信正确样本。本次结果只能支持该新方向的动机，不能作为绕过门槛继续旧RCR的依据。

## 2026-09-18：关闭当前ICAQ上的ADPE训练，转向互补视觉特征审计

D39/D40中ICAQ在hard error上的正margin仅6.0504%/5.7341%，占全部样本3.3603%/2.8314%，
双seed均未达到预注册10%/5%门槛。虽然其margin均值优于Troika原属性分支，证明存在弱
互补信号，但不足以支撑在同一CLIP patch空间上继续训练ADPE，也不支持继续扫描Query、
patch数量或融合权重。

下一阶段只允许验证冻结互补视觉骨干是否能纠正材质属性混淆，先做属性线性探针和错误
互补性审计，不直接接入完整CZSL模型。只有两个seed都显示稳定互补，再设计CLIP与互补
patch特征的融合方案。
