# 本地低学习率 UPerNet：NAS 读取故障恢复

2026-11-17 15:01（北京时间），第 9 轮约 250/801 批次后，DataLoader 在读取 Moon_6 和 Moon_5 原图时遇到 `OSError: [Errno 5] Input/output error`，两卡训练退出。第 20 轮停止控制未触发；CUDA OOM 不是本次日志中的退出原因。

可靠检查点为第 8 轮，验证 mIoU 97.9599%，`last.pt` 的模型、AdamW、poly 调度器及两卡 RNG/加载器状态均可读取；已完成更新 6408 次。第 9 轮未提交的更新需要重跑。

恢复采用原冻结代码、原配置、原训练清单、同样物理 GPU 3/4，从 `last.pt` 接续。训练6401张和验证1601张的数据全部缓存到本地，共16006个文件（图像、标签及两个来源清单），约13.15 GiB。每个文件记录SHA256并验证本地副本，PNG验证完整性；缓存中的两份数据集identity与检查点完全一致。

缓存位置：`/home/lzc/workspace/fenge_vit/data_cache/random80_original6401_1601/`。只在训练进程的私有挂载空间将该缓存只读映射到 `/home/nas`；全机NAS挂载未修改。主机其它任务继续使用原NAS。第20轮控制器重新绑定到恢复后的进程；每5轮评测策略保持。

详见本目录 `recovery.json`、`verification.json`、`stop_controller_launch.json`，以及缓存的 `complete.json` 和 `resume_verification.json`。
