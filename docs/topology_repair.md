# VLM Road Graph Topology Repair

这是一个与 `overhead_ssl` 解耦的可审计原型。安装仓库后可运行：

```bash
python scripts/build_topology_candidates.py path/to/gemini_graph.json \
  --radius 128 --top-k 3 --out data/candidates/scene.json
streamlit run src/topology_repair/annotation_app.py -- \
  --cases data/candidates/scene.json --labels data/labels/annotations.json
```

`RoadGraph.from_dict` 会重建无向邻接关系、去重重复边并重新计算连通分量/端点；不会修改输入 JSON。候选仅从非主分量端点出发，支持外部回调提供 MaGRoad 图/热图桥接支持。`derive_action` 先检查组件真实性，因此 FALSE + CONNECT 始终是 `DELETE_COMPONENT`。

标注保存为一行一个候选的 JSON 数组。`UNCERTAIN` 样本保留在数据集但不会进入监督训练。`baselines.train_two_stage` 使用组件真实性和连接关系两个独立分类器；场景切分由 `scene_split` 完成，避免同一卫星图像泄漏到多个 split。
