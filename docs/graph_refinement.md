# Conservative MaGRoad graph refinement

The graph utilities are independent of encoders and inference. Install `networkx`, `numpy`, `scipy`, `shapely>=2`, and `Pillow`; the project dependencies include them.

```bash
python scripts/graph_refine.py --graph /path/to/raw.json --image /path/to/original.png --probability /path/to/probability.png --output outputs/graph_refinement/scene --parallel-distance 12 --parallel-angle 12 --parallel-overlap .6 --junction-radius 10 --snap-distance 20 --snap-angle 25 --max-gap-distance 35
python scripts/run_refinement_sweep.py --graph /path/to/raw.json --probability /path/to/probability.png --output outputs/graph_refinement/scene
python scripts/evaluate_navigation_graph.py --raw /path/to/raw.json --refined outputs/graph_refinement/scene/refined_graph.json --pairs /path/to/manual_pairs.json --image /path/to/original.png --output outputs/graph_refinement/scene/astar_results
python -m pytest tests/test_graph_refine.py -q
```

Input MaGRoad schema must declare `coordinate_order: "row,column"` with `nodes_rc`, index-pair `edges`, and `inference_shape: [height,width]`. New graph schema declares `coordinate_order: "x,y"`, `units: "inference_pixel"`, nodes `{id,x,y}`, edges `{u,v,geometry,length,confidence?}`, and metadata including inference/original shapes. Polyline geometry is a list of `[x,y]` vertices. Raw files are never written. Only newly exported graphs use the normalized schema.

All distances use inference pixels, angles degrees, scalar probability thresholds/ratios range 0–1. The image is resized explicitly to the graph frame. Probability must already match that frame: single-channel uint8 PNG divided by 255 or float NPY in [0,1]. RGB heatmaps are rejected. Optional image is visualization only; no semantic obstacle detector is inferred from RGB. Without probability, endpoint bridging and spur pruning abstain; duplicate paths can still be consolidated using confidence or topology/length.

`--help` lists configurable thresholds and weights. `config_used.json` records all settings. Gap distance is restricted to at most 40 px in this first version. Parallel consolidation operates on near-straight maximal degree-2 chains and reattaches redundant-path endpoint branches. Junction contraction requires an existing short edge and multiple incident direction families. Snapping adds a short bridge without moving endpoints. Gap candidates must face each other, have sufficient mean and p10 evidence, lack long low-probability intervals, avoid new non-adjacent edge intersections, and be in different components. Accepted/rejected scores are saved for inspection. This intentionally misses genuine loops and some T-junction repairs.

A* pair schema:

```json
[{"id":"route_001","start":[123.0,456.0],"goal":[789.0,321.0]}]
```

Coordinates are inference `[x,y]`, even when the raw graph is rc. For the current XJTLU image, original coordinates must be divided by 2. Pair IDs must be safe filenames. The evaluator projects both points to their nearest edges and temporarily splits those edges, including when both points hit the same edge. Default cost is polyline length. `--uncertainty-lambda` enables an optional nonnegative uncertainty penalty; default 0. Without pairs, the evaluator creates an empty template plus an explicitly named format example; it does not invent evaluation routes.

Stage graphs, overlays, debug images, before/after metrics, all component details and scores are saved. The seven-run sweep varies one parameter at a time rather than claiming a full factorial evaluation. Its baseline max gap is 30 px; the normal refinement default is 35 px.

Current experiment and limitations: `outputs/graph_refinement/xjtlu_1/README_experiment.md`. Outputs are experimental navigation candidates, not a guarantee of a globally connected or semantically valid road network.
