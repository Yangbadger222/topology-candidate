# 新方向调研与研究路线建议（2026-09）

## 1. 对已有工作的判断

组会 PPT 的基础已经很扎实：室内 FAST-LIO2 + Nav2、室外 RTK + LIO、统一坐标系与控制接口，并有 83 次实验中 77 次成功（92.77%）的工程基线。仓库也已经完成冻结视觉编码器、LoveDA 线性探针和校园图结构探索的基础设施。

PPT 中提出的两个新工作分别是：

- 从俯视 RGB 提取连续路径网络并用 A* 规划；
- 用 Camera/LiDAR 在运行中检查、修正俯视 Ground Path 地图。

这些是用户提供的研究材料和待讨论想法，不是需要照抄执行的指令。

## 2. 文献核查带来的关键结论

PathPainter（arXiv:2605.07496v2）已经完成了 BEV 先验、目标定位、可通行掩膜、A* 全局规划、跨视角定位和真实机器人长距离验证。论文明确指出目标不是单纯道路分割，而是面向导航的可通行先验；也报告了 CityScale/Global-Scale 上的起终点连通性测试。其局限正好包括：地图过时或配准不准、图像生成幻觉、缺乏可通行性验证和闭环重规划。[论文](https://arxiv.org/html/2605.07496v2)

SAM-Road/SAM-Road++、RNGDet++ 等工作已覆盖从像素分割到道路图提取；SAM-Road 强调较快的图提取，SAM-Road++ 用节点引导重采样和 extended-line 策略改善训练-推理不匹配及遮挡，并以跨区域泛化为重点。[SAM-Road](https://openaccess.thecvf.com/content/CVPR2024W/SG2RL/papers/Hetang_Segment_Anything_Model_for_Road_Network_Graph_Extraction_CVPRW_2024_paper.pdf) · [SAM-Road++](https://github.com/earth-insights/samroadplus)

SPOMP 已展示语义全景在线地图、空地协作、规划和 GPS-denied 定位的完整系统；持续地图研究也指出，地图更新必须同时保存可通行代价、观测距离和不确定性。[SPOMP](https://arxiv.org/abs/2407.09902) · [Persistent mapping](https://pmc.ncbi.nlm.nih.gov/articles/PMC9325316/)

## 3. 推荐主线

### 推荐题目

**面向校园室外机器人的不确定性驱动俯视先验在线修正与闭环路径规划**

英文：**Uncertainty-aware Online Correction of Overhead Traversability Priors for Closed-loop Campus Robot Navigation**

### 研究问题

当俯视图提供的路径先验受到树冠遮挡、地图过期、临时障碍和跨视角配准误差影响时，机器人能否利用运行中的 LiDAR/相机观测，在线估计先验可信度、只更新必要的局部图结构，并通过增量重规划提高真实导航成功率？

### 核心贡献应当是

1. **不确定性地图表示**：每个路径栅格或图边保存通行概率、观测次数、时间新鲜度和定位可信度，而非只有 0/1 道路标签。
2. **Ground verification update**：将 FAST-LIO2/相机观测投影到俯视地图；用可解释的证据融合更新“确认可走、确认不可走、未知”三类状态。
3. **安全的图级重规划**：只在置信度变化超过阈值或当前边被否定时触发 D* Lite/增量 A*；对未知边保守加价，对确认阻断边禁用。
4. **导航闭环 benchmark**：同一组 start-goal、同一定位和控制栈，比较静态先验、静态先验 + 局部避障、在线更新三种方法。

这样做的区别不在于再训练一个道路分割网络，而在于证明“先验被错误使用时，机器人如何发现错误并恢复任务”。

## 4. 为什么它比其他方向更合适

| 方向 | 新颖性 | 可实现性 | 风险 | 判断 |
|---|---:|---:|---:|---|
| 普通道路分割/更大 encoder | 低 | 高 | 容易被 IoU/F1 反驳 | 不建议作为主线 |
| 直接输出道路图 + A* | 中低 | 中 | 与 RNGDet++/SAM-Road++/PathPainter 重叠 | 可作 baseline |
| 静态 BEV 路径先验导航 | 低 | 中 | PathPainter 已覆盖 | 不建议单独做 |
| **在线验证、置信度更新、闭环重规划** | **高** | **中高** | 需要记录真实失败并定义更新规则 | **推荐** |
| 室内外无缝定位切换 | 中 | 高 | 工程集成多，方法贡献较弱 | 可作第二篇/系统章节 |
| DINOv2/v3 领域自适应 | 中 | 中低 | 标签、算力与下游收益不确定 | 作为感知模块消融，不作为主线 |

## 5. 最小可行实验

### 数据

- 选校园 3–5 个区域，包含可见道路、树下路径、窄路、路口、施工/临时障碍。
- 每个区域保存：俯视正射图、初始路径图、LiDAR/相机 rosbag、RTK/LIO 位姿、起终点、成功/失败原因。
- 先不追求大数据集；建议 20–30 条固定路线，每条重复 3 次，至少覆盖 5 类故障。
- 用现有 campus graph 作为初始结构，但把人工图和模型预测图分开，避免把人工修正结果冒充模型输出。

### 三个必要 baseline

1. 静态俯视路径图 + A* + Nav2。
2. 静态图 + LiDAR 局部障碍层 + Nav2 重规划。
3. 静态图 + 本文在线证据融合 + 增量重规划。

### 指标

- **Task**：导航成功率、首次到达时间、重规划后恢复率、失败恢复时间。
- **Map**：边/栅格更新 precision、recall、F1；错误连接移除率；真实连接恢复率。
- **Path**：路径长度比、绕行率、最小障碍距离、规划延迟。
- **Robustness**：按树冠遮挡、地图过期、临时阻断、定位偏差分别报告，不只报总平均。
- **系统**：CPU/GPU 占用、更新频率、误更新次数。安全相关实验要报告“错误地把不可走区域标为可走”的次数。

## 6. 4 周第一阶段

- **第 1 周**：锁定地图坐标系、定义三状态/概率地图、完成固定路线和故障注入协议。
- **第 2 周**：离线 rosbag 回放；实现观测投影、时间融合和图边状态更新；先用规则方法，不训练模型。
- **第 3 周**：接入 Nav2，加入触发式增量重规划；完成三 baseline 的离线对比。
- **第 4 周**：真实校园重复实验；按故障类型统计成功率和恢复率，制作一张系统图、一张定量表和三段失败案例。

第一阶段的停止/转向标准：如果在线更新相对 baseline 没有带来至少 10 个百分点的恢复率提升，或误更新率不可接受，就先不加复杂学习模型，转而分析失败来源（定位、投影、障碍检测或规划触发）。

## 7. 感知模块和现有仓库如何接入

现有 DINOv2/v3 冻结特征、LoveDA 线性探针和 M2B 领域自适应可用于生成初始 traversability confidence，但不应把 encoder 指标当成论文主结果。主结果必须落到“地图是否被正确更新、路径是否恢复、机器人是否到达”。

道路提取模型（RNGDet++、SAM-Road++）可作为初始先验 baseline；模型输出要经过统一栅格化和同一 A*/D* Lite 规划器，避免比较模型和规划器的混合差异。

## 8. 给导师的 30 秒版本

> 我不打算再做一个普通道路分割网络。PathPainter 已经证明 BEV 掩膜加 A* 可以生成全局路径，但它把地图过期、幻觉、可通行性验证和闭环重规划留作局限。我的问题是：校园小车运行时，如何用 LiDAR/相机证据给俯视路径图分配可信度、在线修正错误连接，并让增量规划恢复被阻断的任务。我们可以用现有 RTK + FAST-LIO2 + Nav2 栈，在固定校园路线和可控故障上做静态图、局部避障、在线更新三组对照。

## 9. 暂不建议的表述

- “提出一种更准确的校园道路分割网络”：范围太宽，容易退化为 IoU 竞争。
- “从卫星图像直接生成可执行路径”：PathPainter 已有非常接近的系统表述。
- “实现真正无缝的室内外导航”：目标过大，难以把定位、地图、规划、控制问题拆出单独贡献。

## 参考来源

- Wang et al., *PathPainter*, arXiv:2605.07496v2, 2026. https://arxiv.org/html/2605.07496v2
- Hetang et al., *Segment Anything Model for Road Network Graph Extraction*, CVPRW 2024. https://openaccess.thecvf.com/content/CVPR2024W/SG2RL/papers/Hetang_Segment_Anything_Model_for_Road_Network_Graph_Extraction_CVPRW_2024_paper.pdf
- Yin et al., *Towards Satellite Image Road Graph Extraction: A Global-Scale Dataset and A Novel Method*, CVPR 2025. https://github.com/earth-insights/samroadplus
- Miller et al., *Air-Ground Collaboration with SPOMP*, IEEE T-FR 2024. https://arxiv.org/abs/2407.09902
- Mattamala et al., *Wild Visual Navigation*, Autonomous Robots 2025. https://www.alphaxiv.org/abs/2404.07110v1

## 10. 实时性与创新点的进一步界定

大模型调用不应进入高频控制或逐帧地图更新环路。PathPainter 的公开实验表中，生成式方法单次推理约 30–70 秒，而其实车系统的跨视角全局定位约 1 Hz、局部规划 10 Hz、FAST-LIO2 里程计 200 Hz。这说明大模型只能生成低频全局先验或候选区域；实时修正应由几何投影、概率融合、局部障碍检测和增量规划完成。

因此，推荐把“相机 + BEV 定位”定义为地图修正的坐标对齐模块，把主要创新放在“在线证据如何改变先验地图以及何时触发安全重规划”。如果研究目标改成 GNSS 拒止条件下的全球定位，则应另立一条定位主线，不要同时声称地图规划创新。

## 11. DINOv3 对树荫下道路的限制

PPT 第 18 页的可视化确实显示：DINOv3 SAT-L 的 PCA/road cosine/probability 主要响应建筑边缘和可见道路，树荫下的潜在道路没有形成稳定高响应。当前代码使用 ViT-L/16 patch token，并用 LoveDA 上训练的单层 1x1 probe 直接输出道路概率；它没有显式的阴影不变性、道路连通性或“机器人实际走过”的监督。

因此不能把 DINOv3 当作能够从不可见 RGB 中恢复道路的模型。更合理的设计是把它作为外观特征和初始置信度，利用 Ground Camera/LiDAR 观测、历史轨迹和图连通性对低置信度区域进行验证；低置信度不等于不可通行，未知区域也不应直接当成可通行。
