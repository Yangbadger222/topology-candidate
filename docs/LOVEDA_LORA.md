# MaGRoad LoveDA Road-only LoRA

实验代码位于 [experiments/magroad_loveda](../experiments/magroad_loveda/docs/LOVEDA_LORA.md)，部署到远端 `/home/badger/sam-inference/MaGRoad/magroad_loveda/`。

本阶段严格训练 Road-only Q/V LoRA（A），然后 LoRA + Map Decoder（B），不运行/训练 TopoNet，A* 仅评估接口与合成测试。完整说明、数据路径、CLI、显存实测、checkpoint 与决策门见上方文档。

本地结果报告：[LOVEDA_LORA_REPORT.md](../outputs/loveda_lora/LOVEDA_LORA_REPORT.md)。完整训练尚未结束时，报告明确显示进行中；远端串行任务结束后，同步进程更新该报告与 comparison 图片。不要把 smoke test 的四张 Val 指标当成完整实验指标。
