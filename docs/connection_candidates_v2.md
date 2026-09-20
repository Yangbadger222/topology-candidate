# Connection Candidate Generator V2

This stage produces proposals only. It never adds edges to a graph, trains a model, or assigns road-validity labels. Source raw MaGRoad row,column coordinates are converted by `graph_utils.load_graph` to x,y inference pixels. Scalar road PNG is divided by 255; this quantized model prediction is not semantic truth.

## Commands

Run from the repository root using Python with numpy, scipy, shapely, networkx and Pillow:

```sh
python3 scripts/candidate_generator_v2.py --graph outputs/MaGRoad/xjtlu_1_globalscale/xjtlu_1_graph.json --probability outputs/MaGRoad/xjtlu_1_globalscale/xjtlu_1_graph_pipeline_probability.png --output outputs/connection_candidates_v2/xjtlu_1
python3 scripts/evaluate_candidate_coverage.py --graph outputs/MaGRoad/xjtlu_1_globalscale/xjtlu_1_graph.json --candidates outputs/connection_candidates_v2/xjtlu_1/all_candidates.json --baseline outputs/learned_graph_refiner_v1/astar/manual_pairs_v1/results.json --output outputs/connection_candidates_v2/xjtlu_1
python3 scripts/visualize_connection_candidates.py --graph outputs/MaGRoad/xjtlu_1_globalscale/xjtlu_1_graph.json --probability outputs/MaGRoad/xjtlu_1_globalscale/xjtlu_1_graph_pipeline_probability.png --image /Users/badger/Desktop/xjtlu_1.png --candidates outputs/connection_candidates_v2/xjtlu_1/all_candidates.json --baseline outputs/learned_graph_refiner_v1/astar/manual_pairs_v1/results.json --coverage outputs/connection_candidates_v2/xjtlu_1/pair_candidate_coverage.json --output outputs/connection_candidates_v2/xjtlu_1
python3 scripts/run_candidate_sweep.py --graph outputs/MaGRoad/xjtlu_1_globalscale/xjtlu_1_graph.json --probability outputs/MaGRoad/xjtlu_1_globalscale/xjtlu_1_graph_pipeline_probability.png --baseline outputs/learned_graph_refiner_v1/astar/manual_pairs_v1/results.json --output outputs/connection_candidates_v2/xjtlu_1
```

`--help` lists all radius, angle, probability-cost, gate and budget parameters. The seven-run sweep changes one factor at a time, retaining the default 80/250/250. It diagnoses proposal count and coverage; it does not choose parameters by test navigation success.

## Interpretation

A retained proposal satisfies `status == proposed` and `budget_excluded == false`. JSON retains hard-gate failures and budget exclusions for auditing. No candidate has a GT label. A degree-two edge boundary remains a valid edge attachment; projections near endpoints/junctions resolve to the appropriate node candidate. KD-tree and STRtree prune local searches. Endpoint tangent uses PCA on a path followed inward up to 30 pixels, stopping at junctions.

Components are indexed by connected-component enumeration on the unchanged raw graph; edges use the unchanged graph edge iterator. Inventory and provenance make these indices auditable. Expanded bounding boxes determine neighboring component pairs, capped at ten selections per component; the undirected union can give a component more than ten incident pairs when neighbors select it. Only these local pairs have seed combinations evaluated, with six searches maximum per selected pair at the default budget three. Endpoints are preferred; components without endpoints use extremal boundary nodes. Nearest target edge projections provide endpoint-edge / edge-edge seeds. At most three retained corridors per actual component pair remain after splitting and global deduplication.

Eight-connected A* uses step length times average cost of adjacent pixels; cost is `1 + lambda*(1-p)**gamma + low_penalty*(p<low_threshold)`. Euclidean distance to a target disk is an admissible lower bound. Sources and goals are disks, then actual graph attachment points are appended to the path. These short attachment sections are included in gates/features. Native C++ acceleration compiles to a source-hashed temporary shared library; if unavailable, Python reference A* is used. Every search logs its window size, expanded pixels and runtime. First-run runtime can include compilation.

Raw search polyline drives crossing and probability features; Douglas–Peucker display geometry is separately saved. Simplified segments stay within the chosen epsilon geometrically, which does not guarantee unchanged probability evidence. Review and future acceptance must use the raw path. Intermediate third-component crossings reject the unsplit parent and generate explicit sub-corridors at the crossing positions, each re-gated. Further ambiguous crossings remain rejected.

Local connectors crossing a graph interior are conservatively gated. Corridor crossings are audited rather than automatically rejected unless they ignore a third component. Same-component local candidates remain permitted and explicitly marked. Endpoint budget five is shared across all local types, with both endpoints counted for endpoint-endpoint connections.

Pair coverage measures candidate geometry within 40 pixels of clicked points and proposals joining the exact nearest-edge component pair. Adjacent-component proposals are a weaker separate diagnostic. A direct component pair proposal is not proof of a correct road, a correct route, or improved A*. The 18 failed projections are analyzed only as diagnostics; clicked points never become candidate seeds or training nodes.

GT-free top review cards cannot establish recall or precision. Building edges, river banks, shadows and tracks can receive high model probability. Human review is required before GT matching or classifier training.
