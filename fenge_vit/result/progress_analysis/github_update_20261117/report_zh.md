# 6401／1601固定划分：结果更新

快照时间：2026-11-17T15:56:13+08:00。

当前数据为 Moon_1／3／5／6／7／8 的 **6401 张原图训练＋1601 张固定验证**，不加离线增强。每轮按验证集累计混淆矩阵 mIoU 选 BEST，每5轮及训练结束后完整测试当时的验证 BEST；完成第20轮保存后停止，自动早停关闭，学习率仍沿用启动时的30轮 poly 计划。验证使用1024／768等权水平翻转，后续完整测试使用 **1024原图＋水平翻转（2视图）**。测试卡不足时保留快照排队。

完整测试包含 Moon_2／4／9 的1597／1320／2087张，共5004张，`S=(mIoU_2+mIoU_4+2×mIoU_9)/4`。`weighted_score`是这个加权S；JSON中的`miou`是合并像素后的mIoU，二者不同。测试场景曾参与历史开发和方案选择，以下是已查看测试集上的开发结果。

本轮6401／1601对照中的最高已测记录为 UPerNet 第14轮的 **93.8405%**，位于第15轮周期评测目录；最终验证 BEST 为第20轮，对应完整 S **93.7585%**。这里明确区分“已测最高”和“训练结束时的验证 BEST”。低学习率 UPerNet 目前仅有第5轮完整测试，不能据此判定它的最终上限。

区域校正和细节修正均从原始 DINOv3 ViT-L＋ConvNeXt 预训练开始，是两组独立实验；刚完成第1轮验证，尚无完整2／4／9成绩。验证分数不能当作测试分数，也不能与完成20轮的基线直接判断模块收益。

## 1024＋水平翻转完整测试

|方案|实际权重轮次|Moon_2 mIoU|Moon_4 mIoU|Moon_9 mIoU|加权 S|
|---|---:|---:|---:|---:|---:|
|[UPerNet，骨干5e-6](../../../result/b_adapter_upernet_vit_large_random80_original6401_103_2gpu/periodic_test/epoch_015/test_hflip_1024_metrics.json)|14|95.3825%|96.1529%|91.9133%|**93.8405%**|
|[UPerNet，骨干5e-6](../../../result/b_adapter_upernet_vit_large_random80_original6401_103_2gpu/periodic_test/epoch_020/test_metrics.json)|20|95.4966%|96.3846%|91.5763%|**93.7585%**|
|[UPerNet，骨干2e-6](../../../result/b_adapter_upernet_vit_large_random80_original6401_lowbackbone_local_2gpu/periodic_test/epoch_005/test_metrics.json)|5|95.1797%|95.8072%|91.9826%|**93.7380%**|
|[Mask2Former，骨干2e-6](../../../result/b_adapter_vit_large_random80_original6401_lowbackbone_103_2gpu/periodic_test/epoch_020/test_metrics.json)|20|93.8135%|94.0150%|91.3975%|**92.6559%**|

## 训练进度

|训练中实验|位置|已完成轮次／目标|验证最佳 mIoU|最近完整轮平均 loss|
|---|---|---:|---:|---:|
|[UPerNet，骨干2e-6](../../../result/b_adapter_upernet_vit_large_random80_original6401_lowbackbone_local_2gpu/summary.json)|本机 GPU 3／4|8／20|97.9599%（第8轮）|0.009313|
|[UPerNet＋区域校正，骨干2e-6](../../../result/b_adapter_upernet_ocr_vit_large_random80_original6401_lowbackbone_103_2gpu/summary.json)|103 GPU 4／5|1／20|94.6639%（第1轮）|0.057023|
|[UPerNet＋细节修正，骨干2e-6](../../../result/b_adapter_upernet_detail_vit_large_random80_original6401_lowbackbone_103_2gpu/summary.json)|103 GPU 6／7|1／20|94.9791%（第1轮）|0.057609|

## 完整测试逐类 IoU

|实验／实际轮次|场景|月壤|岩石|撞击坑|山体|天空|
|---|---|---:|---:|---:|---:|---:|
|UPerNet，骨干5e-6／14|Moon_2|99.5182%|91.6690%|88.2034%|97.5803%|99.9416%|
|UPerNet，骨干5e-6／14|Moon_4|99.5346%|91.6987%|90.7244%|98.8660%|99.9412%|
|UPerNet，骨干5e-6／14|Moon_9|98.7798%|87.4277%|77.7637%|95.8488%|99.7465%|
|UPerNet，骨干5e-6／20|Moon_2|99.5224%|91.8262%|88.7210%|97.4707%|99.9426%|
|UPerNet，骨干5e-6／20|Moon_4|99.5580%|92.0682%|91.3895%|98.9646%|99.9425%|
|UPerNet，骨干5e-6／20|Moon_9|98.7501%|87.8111%|76.0229%|95.5199%|99.7776%|
|UPerNet，骨干2e-6／5|Moon_2|99.4797%|90.2472%|88.8976%|97.3478%|99.9261%|
|UPerNet，骨干2e-6／5|Moon_4|99.4771%|90.1804%|90.7924%|98.6631%|99.9228%|
|UPerNet，骨干2e-6／5|Moon_9|98.7498%|86.6422%|79.8481%|94.9566%|99.7163%|
|Mask2Former，骨干2e-6／20|Moon_2|99.3102%|87.6269%|87.6268%|94.8560%|99.6479%|
|Mask2Former，骨干2e-6／20|Moon_4|99.2184%|87.6921%|87.3065%|96.1970%|99.6606%|
|Mask2Former，骨干2e-6／20|Moon_9|98.6595%|86.1269%|78.3239%|94.4337%|99.4437%|

## 已完成的全部周期评测

四视图历史成绩单独标注；同一个检查点的结束复测／复用可能出现两行，不代表两次独立训练。

|实验|实际轮次|TTA|加权S|来源|
|---|---:|---|---:|---|
|UPerNet，骨干5e-6|5|multiscale_hflip|93.6301%|[JSON](../../../result/b_adapter_upernet_vit_large_random80_original6401_103_2gpu/periodic_test/epoch_005/test_metrics.json)|
|UPerNet，骨干5e-6|10|multiscale_hflip|93.4313%|[JSON](../../../result/b_adapter_upernet_vit_large_random80_original6401_103_2gpu/periodic_test/epoch_010/test_metrics.json)|
|UPerNet，骨干5e-6|14|hflip|93.8405%|[JSON](../../../result/b_adapter_upernet_vit_large_random80_original6401_103_2gpu/periodic_test/epoch_015/test_hflip_1024_metrics.json)|
|UPerNet，骨干5e-6|14|multiscale_hflip|93.5995%|[JSON](../../../result/b_adapter_upernet_vit_large_random80_original6401_103_2gpu/periodic_test/epoch_015/test_metrics.json)|
|UPerNet，骨干5e-6|20|hflip|93.7585%|[JSON](../../../result/b_adapter_upernet_vit_large_random80_original6401_103_2gpu/periodic_test/epoch_020/test_metrics.json)|
|UPerNet，骨干5e-6|20|hflip|93.7585%|[JSON](../../../result/b_adapter_upernet_vit_large_random80_original6401_103_2gpu/periodic_test/final_epoch_020/test_metrics.json)|
|UPerNet，骨干2e-6|5|hflip|93.7380%|[JSON](../../../result/b_adapter_upernet_vit_large_random80_original6401_lowbackbone_local_2gpu/periodic_test/epoch_005/test_metrics.json)|
|Mask2Former，骨干2e-6|5|multiscale_hflip|92.9634%|[JSON](../../../result/b_adapter_vit_large_random80_original6401_lowbackbone_103_2gpu/periodic_test/epoch_005/test_metrics.json)|
|Mask2Former，骨干2e-6|9|multiscale_hflip|92.9252%|[JSON](../../../result/b_adapter_vit_large_random80_original6401_lowbackbone_103_2gpu/periodic_test/epoch_010/test_metrics.json)|
|Mask2Former，骨干2e-6|20|hflip|92.6559%|[JSON](../../../result/b_adapter_vit_large_random80_original6401_lowbackbone_103_2gpu/periodic_test/epoch_020/test_metrics.json)|
|Mask2Former，骨干2e-6|20|hflip|92.6559%|[JSON](../../../result/b_adapter_vit_large_random80_original6401_lowbackbone_103_2gpu/periodic_test/final_epoch_020/test_metrics.json)|
|Mask2Former，骨干5e-6（已停止）|5|multiscale_hflip|92.6237%|[JSON](../../../result/b_adapter_vit_large_random80_original6401_local_2gpu/periodic_test/epoch_005/test_metrics.json)|
|Mask2Former，骨干5e-6（已停止）|10|multiscale_hflip|92.7310%|[JSON](../../../result/b_adapter_vit_large_random80_original6401_local_2gpu/periodic_test/epoch_010/test_metrics.json)|

每个已发布完整测试均检查5004张总数、分场景1597／1320／2087张、五类整数混淆矩阵、逐类别IoU、场景mIoU与1∶1∶2加权S。没有用逐图mIoU平均替代场景计分。

全部训练和验证JSON保留原始内容及来源哈希。新增模型代码是本次工作区版本；各训练任务实际使用的冻结源码身份仍以各自`run_config.json`为准。参数量、CPU一致性检查、双卡完整训练步骤显存检查见[校正头说明](../../../UPERNET_CORRECTION_EXPERIMENTS_20261117.md)。
