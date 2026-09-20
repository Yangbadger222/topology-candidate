# MaGRoad LoveDA graph regression

Inference-only comparison of the WildRoad ViT-B checkpoint, LoveDA encoder LoRA,
and LoveDA encoder-LoRA-plus-decoder checkpoint. The runner preserves the
official WildRoad graph configuration and stops after controlled graph
diagnostics.

Remote execution from the MaGRoad repository root:

```bash
python -u magroad_graph_regression/run_graph_regression.py --phase smoke
python -u magroad_graph_regression/run_graph_regression.py --phase all
```

The runner performs strict checkpoint identity checks and baseline-pipeline
equivalence before any model comparison. It contains no optimizer, backward,
training, skeletonization, A* loss, or topology adaptation code.
