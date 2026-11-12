B0 亮度诊断视觉审阅（固定验证集，none TTA）

已实际查看四组共40图总览，并从每组挑2例，以原始1024网格NPZ裁剪细看，共8例。结论只针对这些目的性选择的例子：错误多贴着坑缘、山岩接触边界或截断目标；目前没有足够视觉证据把投影阴影认定为主要失效原因。

这不是阴影人工标注。没有阴影GT，低亮度仅为亮度分组；“坑内自遮阴”“材质或朝向暗”“疑似投影阴影”等均是待核实的外观描述。固定40图偏向错误/暗区正确条件，这8图也非代表抽样，不能据此估计总体发生率或因果。

B0：epoch29；checkpoint SHA256 `ed1e2096ccd3be31b7ba40ac07c4dbd7849ed1823cb942f4b802706fe1cc6828`；1024×1024、TTA none、固定1601张验证图。数值GT来自原mask，8例均独立核对源RGB、源mask及基准prediction PNG与NPZ逐像素相同。

类别颜色沿用原标注：0月壤紫红、1岩石亮黄、2坑紫、3山体棕、4天空青。错误图橙色为坑→月壤、红色为坑→山体、青色为山体→岩石、紫红色为岩石→山体；黑底为正确。概率图统一固定0..1。

| 分组 | 样本 | 实际可见现象与数值核对 | 暂定外观解释 |
|---|---|---|---|
| dark_crater_missed | Moon_6/1693382863905625088 | GT 坑只在画面左侧留下一小块楔形截断区域，预测为月壤；RGB 对应灰暗有纹理斜面。无法从被截断的局部判定投影阴影或完整坑形。周围可见的其他暗凹地在 GT 中为月壤，不能计作坑漏检。 100% 坑漏分仅涉及整图 244 个 GT 坑像素，不等于一个大坑整体消失。 所选误分 228px，其中 135px 位于任一GT类别边界3px范围内。 | 无法判定（暂定） |
| dark_crater_missed | Moon_3/1693290403749172224 | 可见完整的大坑碗状明暗变化，右侧较暗；GT 坑主体和预测几乎重合，P(crater) 在主体连续且较高。坑到土错误主要是下缘细带，无法将这些边界像素的暗认定为投影阴影。坑内自遮阴只是外观假说。 明显暗部的大坑主体仍保留，整图坑召回率约 99.48%。 所选误分 115px，其中 115px 位于任一GT类别边界3px范围内。 | 坑内自遮阴；无法判定（暂定） |
| bright_crater_missed | Moon_6/1693382836587804160 | GT 坑是远处横向很薄的一条区域，预测保留其主体但轮廓厚度和局部范围不同。所选亮像素错误落在薄坑缘；没有看到大面积亮坑内部被整块改成月壤。 高亮处同样有边界错误；整图坑 IoU 还受预测面积扩张影响，不能只按漏分解释。 所选误分 92px，其中 92px 位于任一GT类别边界3px范围内。 | 无法判定（暂定） |
| bright_crater_missed | Moon_8/1693502043836106240 | 远处 GT 坑呈扁平细条，预测仍有对应连续坑条，主要误差沿上下缘。RGB 的前景暗坑状纹理多数属于 GT 月壤，不用于判定坑漏检。 所选亮区有误分但未见整块大坑消失，说明边界与远距离尺度也需要区分。 所选误分 205px，其中 205px 位于任一GT类别边界3px范围内。 | 无法判定（暂定） |
| dark_mountain_rock_confusion | Moon_5/1693360213595874048 | 山体右端与岩石接触处有很暗的面，GT 岩石在画面右边缘伸出的小片被接到山体；方向图为岩石到山体的紫红色片。暗面较平坦，亮度无法独立分离材质、朝向与遮阴，且同图亮区亦有山岩互混。 暗处同方向错误 521px，亮处也有 300px；不能称该混淆仅发生在暗处。 所选误分 521px，其中 301px 位于任一GT类别边界3px范围内。 | 材质或朝向暗；无法判定（暂定） |
| dark_mountain_rock_confusion | Moon_1/1693274798167865600 | 近景大岩石有明显明暗面，左方地面有疑似投影阴影；GT 岩石主体及其暗面大体保留。山岩互混主要是轮廓附近零星细线，不能把邻近地面暗影当成整块岩石失效的证据。 同图山岩互混暗 67px、亮 462px；亮面也有同方向混淆。 所选误分 67px，其中 67px 位于任一GT类别边界3px范围内。 | 材质或朝向暗；疑似投影阴影（暂定） |
| dark_correct | Moon_7/1693385394043741696 | 大坑内有宽而连续的暗带，外观符合疑似坑内自遮阴；GT 与预测保留完整坑主体，暗带在连续 P(crater) 图中仍是高概率坑。错误主要是坑边缘细线。 大面积疑似自遮阴暗部仍正确；整图 223884 个 GT 坑像素仅漏 390 个，召回约 99.83%。 | 坑内自遮阴（暂定） |
| dark_correct | Moon_1/1693274843058784000 | 岩石暗面与地面三角形暗区都清楚可见；后者外观疑似岩石投影阴影。预测保留岩石的明暗两面和主体轮廓，投影到地面的暗区仍按 GT 月壤处理，未被误当岩石或山体。 暗岩石并非必然被错分；所选暗目标正确像素 107984 个。 | 材质或朝向暗；疑似投影阴影（暂定） |

八例原始NPZ裁剪均未更改曝光；ROI只为显示，模型推理仍是原始1024尺寸。ROI坐标、原图与GT路径/SHA、概率、例点及每张图完整panel路径见 [visual_review.json](visual_review.json)。

可查看的局部对照：

- [Moon_6/1693382863905625088 — dark_crater_missed](/home/lzc/workspace/fenge_result/shadow_b0_random80_epoch029/visual_review/dark_crater_missed_Moon_6_1693382863905625088_roi.png)；原始 [panel](/home/lzc/workspace/fenge_result/shadow_b0_random80_epoch029/val_none/panels/dark_crater_missed/Moon_6/1693382863905625088/panel.jpg)；[NPZ](/home/lzc/workspace/fenge_result/shadow_b0_random80_epoch029/val_none/panels/dark_crater_missed/Moon_6/1693382863905625088/probabilities.npz)。
- [Moon_3/1693290403749172224 — dark_crater_missed](/home/lzc/workspace/fenge_result/shadow_b0_random80_epoch029/visual_review/dark_crater_missed_Moon_3_1693290403749172224_roi.png)；原始 [panel](/home/lzc/workspace/fenge_result/shadow_b0_random80_epoch029/val_none/panels/dark_crater_missed/Moon_3/1693290403749172224/panel.jpg)；[NPZ](/home/lzc/workspace/fenge_result/shadow_b0_random80_epoch029/val_none/panels/dark_crater_missed/Moon_3/1693290403749172224/probabilities.npz)。
- [Moon_6/1693382836587804160 — bright_crater_missed](/home/lzc/workspace/fenge_result/shadow_b0_random80_epoch029/visual_review/bright_crater_missed_Moon_6_1693382836587804160_roi.png)；原始 [panel](/home/lzc/workspace/fenge_result/shadow_b0_random80_epoch029/val_none/panels/bright_crater_missed/Moon_6/1693382836587804160/panel.jpg)；[NPZ](/home/lzc/workspace/fenge_result/shadow_b0_random80_epoch029/val_none/panels/bright_crater_missed/Moon_6/1693382836587804160/probabilities.npz)。
- [Moon_8/1693502043836106240 — bright_crater_missed](/home/lzc/workspace/fenge_result/shadow_b0_random80_epoch029/visual_review/bright_crater_missed_Moon_8_1693502043836106240_roi.png)；原始 [panel](/home/lzc/workspace/fenge_result/shadow_b0_random80_epoch029/val_none/panels/bright_crater_missed/Moon_8/1693502043836106240/panel.jpg)；[NPZ](/home/lzc/workspace/fenge_result/shadow_b0_random80_epoch029/val_none/panels/bright_crater_missed/Moon_8/1693502043836106240/probabilities.npz)。
- [Moon_5/1693360213595874048 — dark_mountain_rock_confusion](/home/lzc/workspace/fenge_result/shadow_b0_random80_epoch029/visual_review/dark_mountain_rock_confusion_Moon_5_1693360213595874048_roi.png)；原始 [panel](/home/lzc/workspace/fenge_result/shadow_b0_random80_epoch029/val_none/panels/dark_mountain_rock_confusion/Moon_5/1693360213595874048/panel.jpg)；[NPZ](/home/lzc/workspace/fenge_result/shadow_b0_random80_epoch029/val_none/panels/dark_mountain_rock_confusion/Moon_5/1693360213595874048/probabilities.npz)。
- [Moon_1/1693274798167865600 — dark_mountain_rock_confusion](/home/lzc/workspace/fenge_result/shadow_b0_random80_epoch029/visual_review/dark_mountain_rock_confusion_Moon_1_1693274798167865600_roi.png)；原始 [panel](/home/lzc/workspace/fenge_result/shadow_b0_random80_epoch029/val_none/panels/dark_mountain_rock_confusion/Moon_1/1693274798167865600/panel.jpg)；[NPZ](/home/lzc/workspace/fenge_result/shadow_b0_random80_epoch029/val_none/panels/dark_mountain_rock_confusion/Moon_1/1693274798167865600/probabilities.npz)。
- [Moon_7/1693385394043741696 — dark_correct](/home/lzc/workspace/fenge_result/shadow_b0_random80_epoch029/visual_review/dark_correct_Moon_7_1693385394043741696_roi.png)；原始 [panel](/home/lzc/workspace/fenge_result/shadow_b0_random80_epoch029/val_none/panels/dark_correct/Moon_7/1693385394043741696/panel.jpg)；[NPZ](/home/lzc/workspace/fenge_result/shadow_b0_random80_epoch029/val_none/panels/dark_correct/Moon_7/1693385394043741696/probabilities.npz)。
- [Moon_1/1693274843058784000 — dark_correct](/home/lzc/workspace/fenge_result/shadow_b0_random80_epoch029/visual_review/dark_correct_Moon_1_1693274843058784000_roi.png)；原始 [panel](/home/lzc/workspace/fenge_result/shadow_b0_random80_epoch029/val_none/panels/dark_correct/Moon_1/1693274843058784000/panel.jpg)；[NPZ](/home/lzc/workspace/fenge_result/shadow_b0_random80_epoch029/val_none/panels/dark_correct/Moon_1/1693274843058784000/probabilities.npz)。

审阅中需要保留的反例与限制：

- Moon_6/1693382863905625088 的“100%漏分”是244像素的左缘截断小坑；不能解读为整块大坑消失。
- Moon_3/1693290403749172224 的大坑主体保留，坑召回约99.48%；Moon_7/1693385394043741696 的疑似自遮阴大坑召回约99.83%。
- 两张亮坑例都保留连续的远处细坑条，错误主要沿边缘；当前细看例子未见完整大亮坑的整块漏检。
- 两例山岩互混在暗、亮像素均存在。相同错误的亮像素分别300、462；不能称其只发生在阴暗区域。
- 6个错误案例所选误分像素中没有Y≤0.02的近黑像素；Moon_5例虽有182个局部std≤0.005的误分像素，其亮度约0.086，并非规则中的近黑。黑天空不计作地表近黑。
- 原图中看起来像小坑或投影阴影的纹理，若GT为月壤，不计作漏坑。未根据RGB主观增补GT。
- 本审阅不启动模块实验。是否进入后续实验，由主任务结合全量同图分层统计和边界、尺度等混杂检查决定。
