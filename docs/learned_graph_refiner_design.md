# learned_graph_refiner_v1：已核实设计与 GT 门槛

## 源码核实（2026-09-16）
实际部署目录 `/home/badger/sam-inference/MaGRoad`；源码快照位于 outputs/learned_graph_refiner_v1/source_audit。Global-Scale config 的 MODEL_NAME=MaGRoad，因此有效实现是 model.py 的 MaGRoad/MaGTopoNet，不是 sam_road_plus_model.py 的 SAMRoadplus。

- model.py: MaGRoad.__init__：ViT-B 内部 embedding 768，neck 输出 256 channels；patch16，512 输入输出 [B,256,32,32]，stride16。map_decoder 四次 ConvTranspose2d 上采样至 [B,2,512,512]；channel0 keypoint、channel1 road。
- infer_masks_and_img_features：RGB float32 0–255、BHWC；按 pixel_mean/std 归一化，返回 encoder embedding、BCHW mask logits、BHWC sigmoid scores。
- BilinearSampler：xy/512*2-1，grid_sample bilinear、align_corners=False、默认 zero padding。
- MaGTopoNet.forward 接收 points、point_features、pairs、pairs_valid、mask_logits。config USE_POINT_FEATURES=False：即使 infer_toponet 计算 node feature，TopoNet 不使用它。USE_PATH_FEATURES/USE_GEOMETRIC_FEATURES/USE_EDGE_BIAS=True。
- geometry：dx/dy、距离、方向 Fourier 共11维→128；原生 path extractor 对 road logits sigmoid、kernel1/3/5平均池化，直线32点采样，每尺度 mean/std/softmin(1-p,tau5)，合计9维→128。它不是 encoder path semantic feature。融合256维 edge token，4层8头 attention，bias来自转向/竞争几何。
- infer_one_img：重叠512patch，短边16、长边23（368tiles），概率均值融合；uint8 导出有1/255量化。随后提取 road/keypoint graph points，邻居半径64、最多16候选；TopoNet threshold .455。可能漏掉超过原候选半径的连接。

## 缓存与新特征
不改模型、不改 checkpoint，strict=True加载，eval+inference_mode，所有参数冻结。重用部署 inferencer 的 patch布局（通过 AST 只加载指定函数，避免其顶层 CLI副作用）。CPU float16保存各tile完整 [256,32,32] embedding 和 [2,512,512] logits，避免跨tile直接拼接破坏patch上下文；全图融合概率float32另存。记录tile origin、输入尺寸、实际tensor shape、stride、RGB/config/checkpoint/source hash、git commit、时间与字节。

样本均匀沿polyline弧长32点；每个点在覆盖它的tile内使用模型 BilinearSampler，选择离tile边界最远者（显式策略，非官方融合）。端点/中点+path mean/std/min构成1536维 encoder组；gap另加abs(A-B)、A*B成为2048维。road统计与原生9维path组成 probability组；几何和图上下文独立为geometry组。原生path只适用于同一tile内直线，跨tile或曲线返回 applicability=False，不伪称完全复用。

## 当前 stop condition
找到远端 overhead_paths/D 的76张1024图，4个地理region，2788路径片段；全为draft。片段没有已审核 junction graph，完整覆盖范围、负例区域和可通行定义尚未验证；XJTLU已审核centerline GT=0。不能把未标区域自动当negative，也不能用MaGRoad预测当GT。

本阶段落地缓存、标注/点击工具、草稿转换、GT验证、region split验证、label及feature dataset生成。训练、阈值选择、learned graph、ablation与正式A*等待人工GT和人工pairs。禁止给A/B研究结论：无有效测试数据不能判定feature足够或不足。
