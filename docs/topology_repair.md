# Gemini Road Graph Topology Repair V2

V2 是一个保守的人工数据制作工具：Gemini 初始图保持只读，只挖掘可疑局部并保存人工标签；当前阶段不训练模型、不使用 Gemini 自己当 GT，也不要求 MaGRoad。

## 输入与构建

每个 scene 只需要卫星 RGB 和 Gemini JSON：

```text
scene_001/satellite.jpg
scene_001/gemini_graph.json
```

```bash
PYTHONPATH=src python scripts/build_topology_annotation_cases.py \
  --image scene_001/satellite.jpg \
  --graph scene_001/gemini_graph.json \
  --scene-id scene_001 \
  --output data/annotation_cases/scene_001
```

输出 `cases.json`、`source_inventory.json` 和四种渲染图。配置可通过 `--config` 指定；默认限制包括 source 节点/长度、最小 branch 长度、搜索半径和 Top-K。

边解析会保留 `coordinates`、`polyline` 或 `geometry.coordinates` 中的完整 polyline；缺失时才退化为节点直线。重复无向 node pair 只保留输入中的第一条 canonical edge，方向统一为较小 node id 到较大 node id。长度、投影和渲染均基于 polyline 弧长/线段。

## Source subgraph 与候选

工具发现两类 source：

- `disconnected_component`：主分量之外且通过大小/长度阈值的组件；
- `leaf_branch`：从 degree-1 endpoint 沿 degree-2 链走到 junction 的 branch。删除 leaf branch 时只删除 branch-owned edges/nodes，attachment junction 保留。

每个 source endpoint 生成 endpoint→endpoint 与 endpoint→polyline edge interior 候选。leaf branch 可以连接到同一 connected component 中、但不属于 source 的 edge。候选只保存几何/拓扑特征；MaGRoad support 如需使用是可选回调，不是输入要求。

## 标注 UI

```bash
streamlit run src/topology_repair/annotation_app.py -- \
  --cases data/annotation_cases/scene_001/cases.json \
  --labels data/labels/scene_001.json
```

一个页面对应一个 source endpoint，而不是一个候选。页面提供 Raw/Overlay Local/Context 四个视图：黄色是 source，青色是周围 Gemini graph，白色 S 是 endpoint，绿色 T1/T2/T3 是候选 target。标注前不显示 heuristic、模型或 MaGRoad 分数。

先回答 source validity：`REAL`、`FALSE`、`UNCERTAIN`。FALSE 直接结束该 source 的连接问题。REAL 时再选择 `T1...T3`、`NO_CONNECTION`、`CORRECT_TARGET_NOT_PROPOSED` 或 `UNCERTAIN`。其中 `CORRECT_TARGET_NOT_PROPOSED` 不等同于 NO_CONNECTION，可用于计算 candidate Top-K recall。Reason codes（如 `ROOFTOP`、`SHADOW`、`TRUE_DEAD_END`）和备注均为可选。

## Schema V2 与动作

标签文件使用 source-level validity 和 endpoint-level selection 分离的结构：

```json
{
  "schema_version": 2,
  "scene_id": "scene_001",
  "source_subgraph_id": "source_00001",
  "source_type": "leaf_branch",
  "component_validity": "REAL",
  "endpoint_labels": {
    "85": {
      "selection": "candidate_00031",
      "derived_action": "ADD_CONNECTION",
      "reason_codes": [],
      "notes": ""
    }
  },
  "annotator": "human",
  "created_at": "...",
  "updated_at": "...",
  "provenance": {"graph_sha256": "...", "image_sha256": "..."}
}
```

动作优先级：`FALSE → DELETE_SOURCE_SUBGRAPH`；`UNCERTAIN component → REVIEW`；REAL + candidate → `ADD_CONNECTION`；REAL + `NO_CONNECTION → KEEP`；REAL + `CORRECT_TARGET_NOT_PROPOSED → CANDIDATE_MISS`；REAL + UNCERTAIN → `REVIEW`。FALSE 永远优先，即便 endpoint selection 未确定。

保存使用 temporary file + flush/fsync + `os.replace`，避免长时间标注后损坏整个 JSON。原始 Gemini JSON 不会被修改。

## 测试

```bash
PYTHONPATH=src pytest -q tests/test_topology_repair.py
```

测试覆盖 curved polyline、弧长、polyline 投影、leaf branch、junction 保留、多 endpoint 标签、候选缺失选项、same-component target、atomic save 和无 MaGRoad 运行。
