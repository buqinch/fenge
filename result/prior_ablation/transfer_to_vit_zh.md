# CNN 先验整理：供后续 ViT 独立消融使用

整理日期：2026-11-13。依据四组 CNN 已完成的 30 轮训练，以及各自最终验证 best 的完整 Moon_2／4／9 测试。本文件整理实现和候选实验，未修改模型或启动训练。

**优先保留高分辨率细节注入作为 ViT 的独立候选。** 它在本次 CNN 对照中使加权 S 提高 0.0817 个百分点，但 Moon_9 下降 0.4845 个百分点，收益很小且并不全面。周围关系和当前语义亲和实现的最终 S 均低于对照，不能列为已经验证有效的优化；它们保留为可复核实现。CNN 结果不代表 ViT 已验证收益。

## 1. 最终检查点的实测结果

训练均为固定 8000 张＝4000 原图＋2000 坑增强＋2000 山岩增强，完整验证为 1601 张，验证使用 1024 单视图、无 TTA。所有检查点按各自验证 mIoU 的首次严格最高值选择。下表完整测试均为 5004 张：Moon_2 1597、Moon_4 1320、Moon_9 2087；1024／768 等权水平翻转共 4 视图，类别分数系数均为 1。

`S = (Moon_2 mIoU + Moon_4 mIoU + 2 × Moon_9 mIoU) / 4`。表内成绩由逐场景整数混淆矩阵独立复算，未使用 pooled mIoU 或逐图 mIoU 的平均值。

|CNN 方案|最终验证 best 轮次|Moon_2 mIoU|Moon_4 mIoU|Moon_9 mIoU|加权 S|S 相对对照（百分点）|
|---|---:|---:|---:|---:|---:|---:|
|对照|28|93.6695%|93.5634%|86.3247%|89.9706%|+0.0000|
|细节注入|30|94.2704%|94.2583%|85.8402%|90.0523%|+0.0817|
|周围关系|30|94.0234%|93.9374%|85.7678%|89.8741%|-0.0964|
|语义亲和|28|93.7410%|93.4841%|85.8571%|89.7348%|-0.2357|

细节注入第 11 轮的历史测试 S＝89.9852%；当前最终验证 best 是第 30 轮，S＝90.0523%。第 11 轮只用于追溯，不能取代最终验证选出的检查点，也不能混写为第 30 轮成绩。

下面均为相对对照的 IoU 百分点变化，显示收益来自哪些类别。

|方案|场景|月壤|岩石|撞击坑|山体|天空|
|---|---|---:|---:|---:|---:|---:|
|细节注入|Moon_2|+0.0820|+2.2255|-0.2697|+0.8245|+0.1422|
|细节注入|Moon_4|+0.1562|+2.5952|-0.2302|+0.7673|+0.1860|
|细节注入|Moon_9|+0.0275|+0.2130|-0.6725|-2.0991|+0.1085|
|周围关系|Moon_2|+0.0558|+1.8190|-1.0204|+0.7776|+0.1373|
|周围关系|Moon_4|+0.1071|+1.5402|-0.7399|+0.7928|+0.1699|
|周围关系|Moon_9|+0.0089|-0.0637|-1.6225|-1.1992|+0.0923|
|语义亲和|Moon_2|+0.0019|+0.1545|+0.0307|+0.1648|+0.0054|
|语义亲和|Moon_4|-0.0070|+0.0042|-0.2984|-0.0920|-0.0031|
|语义亲和|Moon_9|-0.0333|-0.6821|-2.2695|+0.6335|+0.0137|

细节注入的主要正变化是 Moon_2／4 岩石 IoU 提高 2.2255／2.5952 个百分点，以及这些场景的山体改善。仅凭类别 IoU 不能进一步声称“小岩石召回”或“边界准确率”已经改善，后续需要面积分组和边界指标。Moon_9 山体 IoU 反而下降 2.0991 个百分点，坑 IoU 下降 0.6725 个百分点。

|方案|Moon_9 坑 IoU|坑精确率|坑召回率|坑召回相对对照（百分点）|
|---|---:|---:|---:|---:|
|对照|58.8131%|84.0776%|66.1847%|+0.0000|
|细节注入|58.1406%|86.2674%|64.0705%|-2.1142|
|周围关系|57.1906%|85.4476%|63.3621%|-2.8226|
|语义亲和|56.5436%|84.8381%|62.8998%|-3.2848|

细节注入在 Moon_9 的坑→月壤由 8,452,590 减至 7,964,869 像素，但坑→山体由 910,567 增至 1,989,490 像素，因此坑整体召回仍下降。月壤→山体也增加 1,290,130 像素。说明不能只看一条错分减少，也不能把视觉边缘变清晰等同于语义改善。

## 2. 哪些模块值得迁移

|优先级|模块|当前判断|后续用途|
|---|---|---|---|
|1|语义引导的高分辨率细节残差|本次 S 小幅提高，Moon_9 有明显退步；尚无重复种子证据|先做 ViT 独立消融，重点监控山体误检和坑→山体|
|2|局部与周围关系残差|最终 S 比对照低 0.0964 个百分点|暂不加入主方案；只有 ViT 错例支持上下文不足时再单独试|
|3|稀疏语义亲和辅助监督|最终 S 比对照低 0.2357 个百分点，Moon_9 坑召回下降|保留诊断和采样实现；当前权重 0.05 不推荐默认迁移|

已淘汰的几何实验不重新包装为有效先验，清理依据见 [既有清理记录](pruned_runs_20261112.json)。高分辨率细节、周围关系和亲和不是一组必须同时开启的组件。

## 3. 已实现的 CNN 接口与参数

以下代码和配置均是实际存在的实现。CNN 相对链接从本目录可直接打开；ViT 链接指向同一 GitHub 仓库的 `fenge_vit/` 源码与配置，实际服务器部署目录为 `/home/lzc/workspace/fenge_vit`。

### 细节注入：保留完整实现作为首选候选

[DetailRefinement 与共享残差实现](../../code/lunarseg/refinement.py)、[模型接入](../../code/lunarseg/model.py)、[当前细节配置](../../config/dual_region_priors_detail_1024.json)。

接口为 `refiner(mask_features, images, semantic_logits, probabilities)`：

|输入|CNN 当前形状／含义|ViT 适配目标|
|---|---|---|
|mask_features|B×128×256×256，FPN 掩码特征|B×256×256×256，Mask2Former pixel decoder 的掩码特征|
|images|B×3×1024×1024，ImageNet 归一化 RGB|沿用 ViT 两骨干实际收到的同一张图|
|semantic_logits|B×5×1024×1024，区域头归一化后的 log 概率|沿用 ViT 现有概率转换得到的 log 概率|
|probabilities|对应的五类概率，用于门控|从既有 log 概率取 exp，不对区域合成证据直接 softmax|

具体通道和连接：

- 输入先按 ImageNet 均值／标准差反归一化并裁到 [0,1]；3×3 卷积 3→8，在原分辨率保留细节，再以步长 2 的 3×3 卷积 8→32 得到 512×512 特征。
- 掩码特征以 1×1 卷积 128→32 投影，再双线性插值到 512×512；与 32 通道细节拼成 64 通道，经过 64→32 的 1×1 和 32→32 的 3×3 卷积融合。ViT 只需将投影输入通道改为 256。
- 修正头为 32→5 的 1×1 卷积；门控将 32 通道融合特征与 5 类初始概率拼接为 37 通道，经 37→16→1 输出 gate logits。中间卷积使用 GroupNorm 和 GELU。
- 仅 5 通道修正和 1 通道门控插值到最终图像尺寸；避免把 128／256 通道掩码特征直接展开到 1024。768 TTA 视图按输入实际尺寸得到 768／384／192 的对应网格，不能写死 512。
- 修正头末层、门控末层权重和偏置均零初始化，初始 Δ＝0、sigmoid(gate logits)＝0.5，修正初始不改变预测；未通过熵阈值限制可修改区域。
- CNN 在独立 CPU 随机流中以 `REFINEMENT_INIT_SEED=20261112` 初始化，保持基础网络初始化不变；`prior_seed=4242` 是亲和采样／初始化种子，不是细节模块初始化种子。

CNN 输出为 `z = log(p0) + sigmoid(gate_logits) * delta`，由现有语义 CE／Dice 监督，并在预测时转为概率。开关 `refinement="detail"`；默认 `"none"`。修正参数使用 `prior_lr=1e-4`，随原 poly 学习率计划衰减；它没有额外的亲和损失，`prior_warmup_updates` 不控制细节残差幅度。

### 周围关系：实现已具备方向和多尺度，不重复堆叠

[ContextRefinement / directional_context](../../code/lunarseg/refinement.py)、[周围关系配置](../../config/dual_region_priors_context_1024.json)。

输入接口与共享残差一致，但当前分支不使用原图或初始概率。128 通道掩码特征先投影到 32，在 1/4 网格计算：9×9 局部均值、33×33 方框去掉 9×9 中心的环带均值，以及上／下 12×9、左／右 9×12 对称条带均值；边缘 replicate 填充，FP32 积分图实现。

拼接 7 组 32 通道：`feature`、`local-feature`、`surround-local`、`up-local`、`down-local`、`left-local`、`right-local`，共 224 通道；经 224→32 和 32→32 混合后预测 5 通道修正与 1 通道门控。两末层同样零初始化。开关 `refinement="context"`，损失沿用现有语义 CE／Dice，无额外手工类别规则。

后续 ViT 已有四阶段 Adapter、可变形多尺度 pixel decoder 和 query attention。上下文模块并非明显缺项；若再试该分支，需说明它的局部对应关系补充了什么，不能把相似上下文结构的叠加本身当作收益。

### 语义亲和：训练专用辅助头

[SemanticAffinity](../../code/lunarseg/priors.py)、[训练接入与优化器](../../code/train.py)、[亲和配置](../../config/dual_region_priors_affinity_1024.json)。

接口 `SemanticAffinity(width=128, projected=32, quota=512, seed=4242)(features, targets, sample_keys, epoch)` 返回辅助损失和采样统计。使用 `model(..., return_features=True)["mask_features"]`；额外权重保存在检查点独立 `affinity` 键，推理不需要该头，也没有亲和传播或预测后平滑。

- 1×1 卷积将 128 通道特征投影为 32；成对描述为两端均值、绝对差、逐元素乘积、归一化相对坐标绝对值，共 98 维，经 Linear 98→64→1 预测同类 logit。
- 原始 1024 标签网格上采样，不先缩小 GT。地表四类每类最多 2048 个锚点，半径 8／16／32／64、八方向，去除反向重复对；天空和 ignore 端点不参与。
- 每图最多 512 对，同类／异类各最多 256；在类别对×半径层内轮转采样，不足不复制。边界点保留，采样由样本标识、epoch、seed 确定。
- 按原图坐标从低分辨率投影特征双线性采样；BCE 先分别对正／负组取均值，再对有效组和图像取均值。当前头随机初始化，不采用细节分支的零输出初始化。
- 损失为 `L_base + 0.05 * min(1,(global_update+1)/1000) * L_affinity`；辅助参数学习率 1e-4，默认 `affinity_weight=0`。当前实现监督类别一致性，不使用物理实例或真实阴影标注。

## 4. ViT 接入前必须处理的两处差异

[ViT 模型概率路径](https://github.com/buqinch/fenge/blob/main/fenge_vit/code/vitseg/model.py)、[Mask2Former 掩码特征](https://github.com/buqinch/fenge/blob/main/fenge_vit/code/vitseg/decoder.py)、[ViT 现有损失](https://github.com/buqinch/fenge/blob/main/fenge_vit/code/vitseg/losses.py)、[推理与 TTA](https://github.com/buqinch/fenge/blob/main/fenge_vit/code/vitseg/metrics.py)、[遮挡 NLL](https://github.com/buqinch/fenge/blob/main/fenge_vit/code/vitseg/masking.py)。

**第一，后段修正必须保持归一化 log 概率接口。** ViT 当前按 `p0=(S+eps)/(sum(S)+5*eps)` 转换区域合成证据；这与 CNN 的 clamp 细节不同，迁移时保留 ViT 既有定义。新增修正后输出应为 `log_p = log_softmax(log_p0 + g*delta, dim=1)`。当前 TTA 直接对 `semantic_logits.exp()` 平均，C 的遮挡损失直接以该值做 NLL；若直接返回 CNN 的未重新归一化残差 z，会破坏这两个调用约定。不能对合成证据 S 任意套 softmax，也不能令旧 `semantic_scores` 与修正后的语义输出含义混淆。

**第二，现有 ViT 普通 Mask2Former 损失不会训练后段修正。** 它只监督 `pred_logits`、`pred_masks` 和各层 `aux_outputs`，不读取最终 `semantic_logits`。只增加 refiner 会导致 A／B 的细化头没有监督梯度；C 也只能在部分遮挡更新中获得相关梯度，不能作为充分接入。

建议独立定义一个后续待实现的完整图语义 NLL 项，保留全部原 Mask2Former 监督：`L = L_M2F + lambda_sem * NLL(log_p,Y)`。权重需预先固定并在验证集评估，不照搬 CNN 总损失数值；可把 0.2 作为待验证起点，但这不是已实现开关或有效超参数。为了分离新增损失与细节分支收益，必须同时设置“原 ViT＋同一个语义 NLL”的对照，不能只给细节组加监督。先不同时加入 Dice、亲和或几何约束；若要加入，两个对照也应保持一致。

Mask2FormerHead 目前未返回内部 `mask_features`，迁移时需显式提供训练／推理可选输出，保留原 query 预测键和辅助监督。四卡入口还要把新增语义损失按全局有效像素数归约，按同一规则记录损失；不能简单平均不等有效像素数的 rank 均值。C 的遮挡支路必须让细节分支收到同一张已遮挡图，禁止从旁路读到未遮挡原图。

优先在 `fenge_vit/code/vitseg/` 内独立实现可选模块，避免跨工程 import CNN 训练模块；复用接口与算法，不载入 CNN 分割权重。开关关闭时旧 ViT 配置／检查点保持可用；开启时新增权重必须显式保存，不能静默丢弃不匹配键。

## 5. 下一轮消融的推荐组织

原有 [A 配置](https://github.com/buqinch/fenge/blob/main/fenge_vit/config/a_fusion.json)、[B 配置](https://github.com/buqinch/fenge/blob/main/fenge_vit/config/b_adapter.json)、[C 配置](https://github.com/buqinch/fenge/blob/main/fenge_vit/config/c_adapter_masking.json) 的含义保持：A＝简单融合，B＝四阶段 Adapter，C＝B＋遮挡监督。后段细节是独立因素，不重新命名 A/B/C 混淆已有结果。

|顺序|后续候选|可回答的问题|
|---|---|---|
|1|B 原配置；B＋完整图语义 NLL；B＋相同 NLL＋detail|先分离新增监督收益，再测细节增益|
|2|C＋相同 NLL；C＋相同 NLL＋detail|在完全一致的遮挡设置下检查细节与遮挡是否互补|
|3|必要时 A＋相同 NLL；A＋相同 NLL＋detail|检查收益是否依赖 Adapter，而非将 CNN 结论外推|
|4|有明确错例依据后再独立 context 或 affinity|避免在尚未确认有效的分支上堆叠多项变化|

这是一份后续实验计划，不表示已经运行或安排新任务。具体比较应保持同一主机／GPU、相同每卡 batch 与累积、相同训练清单／验证集、随机种子、冻结阶段、学习率和最大更新预算。各候选从原始 DINOv3 ViT-S/16＋ConvNeXt-Tiny 预训练开始，公共网络初始化一致，新增模块使用独立随机流；不从已经见过更多样本的 CNN 或 ViT best 继续训练来冒充从头对照。

当前四组 CNN 均完成 30 轮、每轮 1000 更新，共 30000 更新，前 3 轮冻结；但对照／亲和是当前服务器实际 batch 4、累积 2，周围／细节是 103 实际 batch 8、累积 1，且后两组开启激活重计算。有效 batch 都为 8，并不消除 MobileNet BN、按微批计算 Dice、设备／软件及训练轨迹差异。本次单种子差异不能归因于模块本身的稳定收益。四组基础网络初始化哈希一致：`48afac5eac0038e6e4a0aa55b6c472a68cde716b0492943b9181c182687a2740`。

后续若用当前 6000 张配比，整套新对照统一用 4000＋1000＋1000，30 轮为 22500 更新；8000 张配比 30 轮为 30000 更新。不要把不同配比／预算与细节开关同时改变后归因为先验收益。固定 1601 张验证集选 best；继续按每 5 轮完整测试 Moon_2／4／9 的既有节奏记录，测试不参与检查点选择。测试集已被查看，后续结果属于同一已查看测试集上的实验对照。

判定时同时报告 S、三场景 mIoU、五类 IoU、坑→月壤／坑→山体／月壤→坑、月壤→山体以及山岩双向错分；若要主张细节改善，增加固定定义的小岩石和边界统计。先做零初始化前后概率一致性、真实损失到细化头梯度、1024／768 坐标对齐、四卡归约及严格重载检查，再开始新的正式消融。

## 6. 可复核文件

|方案|训练配置|实际运行配置|完整测试|
|---|---|---|---|
|对照|[配置](../../config/dual_region_priors_control_1024.json)|[run_config](../dual_region_priors_control_8000_1024/run_config.json)|[第 28 轮指标](../dual_region_priors_control_8000_1024/test_epoch_028_multiscale_hflip_1024_768_metrics.json)|
|细节注入|[配置](../../config/dual_region_priors_detail_1024.json)|[run_config](../dual_region_priors_detail_8000_1024/run_config.json)|[第 30 轮指标](../dual_region_priors_detail_8000_1024/test_epoch_030_multiscale_hflip_1024_768_metrics.json)|
|周围关系|[配置](../../config/dual_region_priors_context_1024.json)|[run_config](../dual_region_priors_context_8000_1024/run_config.json)|[第 30 轮指标](../dual_region_priors_context_8000_1024/test_epoch_030_multiscale_hflip_1024_768_metrics.json)|
|语义亲和|[配置](../../config/dual_region_priors_affinity_1024.json)|[run_config](../dual_region_priors_affinity_8000_1024/run_config.json)|[第 28 轮指标](../dual_region_priors_affinity_8000_1024/test_epoch_028_multiscale_hflip_1024_768_metrics.json)|

- [细节第 11 轮历史指标](../dual_region_priors_detail_8000_1024/test_epoch_011_multiscale_hflip_1024_768_metrics.json)
- [固定 8000 张选择清单](../selections/random80_4000_2000_2000_seed42_20261112.json)；[固定验证来源池](../selections/random80_4000_2000_2000_pool_20261112.json)
- [现有 CNN 先验检查入口](../../code/verify_priors.py)；[真实网络执行检查](verification_runtime.json)
- [最新 CNN／ViT 完整统计](https://github.com/buqinch/fenge/blob/main/fenge_vit/result/progress_analysis/latest_report_zh.md)

完整测试源文件 SHA256：

|方案|SHA256|
|---|---|
|对照|`6d15a69657e4030cdc76f206833d3ddba21dea50965c6b723f90bb2634f899fd`|
|细节注入|`f19845dbe985225489107fa8fc8e23db61f9de7d62aa6a6bc7d2f9f8994a2ed1`|
|周围关系|`0f220377e9c925395472f228e30b65ce636b7fb76b68e91def4918493bddace5`|
|语义亲和|`d247ad39124fd64337bd79e4b199b4ff8e14dd5445d3de9802bcb44077fa62fd`|
