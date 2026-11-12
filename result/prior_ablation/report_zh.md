# 三种先验与高分辨率细节独立对照

所有组从原始 DINOv3 ConvNeXt-Tiny＋ImageNet MobileNetV3-Large 开始，融合和区域头同种子重新初始化；没有加载第29轮分割权重。
固定8000张训练（4000原图＋2000坑增强＋2000山岩增强），同一1601张验证，1024整图、有效batch 8，验证单视图。默认microbatch 4×累积2；显式覆盖见下表。

|实验|对照|主机|实际batch×累积|重算|状态|完成轮数|最佳轮次|验证mIoU|相对对照百分点|
|---|---|---|---|---|---|---:|---:|---:|---:|
|control|—|Server70|4×2|False|running|28|28|97.4096%|—|
|affinity|control|Server70|4×2|False|running|17|17|97.2188%|-0.1908|
|context|control|Server103|8×1|True|running|11|11|98.0571%|+0.6475|
|detail|control|Server103|8×1|True|running|10|10|98.1394%|+0.7298|

亲和、中心—周围关系和细节注入分别与单视图control比较；几何一致性与同样两视图真值监督的geometry_control比较。
最多30轮并使用相同早停规则；若实际更新数不同，不把最佳分差解释为固定训练预算下的纯模块因果收益。单种子结果尚不能证明稳定提升。
完整类别IoU、精确率、召回率、六场景和混淆方向见[comparison.json](comparison.json)。

**批大小差异限制：** context、detail 按显式计划使用不同的实际batch/累积组合。即使有效batch同为8，MobileNet BatchNorm的统计和更新次数、worker分配造成的水平翻转、随机监督点与数值轨迹仍可能不同；相对4×2对照的分差不能只归因于新增模块。
样本顺序hash仍逐轮强制一致；翻转与样本＋翻转hash逐轮核对，只有显式异batch比较允许报告不一致，具体轮次保存在training_stream_audit。

**实验尚未全部结束，表中为阶段性结果，不据此选择最终方案。**
