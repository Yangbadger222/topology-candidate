# LoveDA Road-only MaGRoad LoRA

Goal: test whether Chinese Google Earth road recall improves by adapting the WildRoad SAM ViT-B encoder. This milestone excludes topology, skeletons, junctions, centerline/clDice, planning losses, and A* training. A* is a tested evaluation interface only.

## Code audit and compatibility

Inspected the actual remote `model.py`, `train.py`, dataset/datamodule, WildRoad and Global-Scale configs, checkpoint tensors, inference path, and SAM encoder. The official LoRA wraps every block's Q/V projections with unscaled `B(A(x))`; A uses Kaiming initialization, B starts at zero, and the original QKV weight/bias retain their names. Only encoder base is frozen by the official boolean. Official optimizer still includes map decoder and TopoNet. WildRoad LR is 1e-3, encoder factor .1. The two-channel ConvTranspose decoder is keypoint=0, road=1. The ordinary forward executes topology; the mask inference path does not. Native 1024 gives positional embedding [1,64,64,768]; Global-Scale 512 is not used here. Existing Lightning training defaults FP16 but does not configure accumulation.

The adapter runner adds explicit modes and independently constructs optimizer groups. A freezes everything except Q/V LoRA (147456 parameters). B additionally trains map decoder (172770 parameters; total 320226). TopoNet is frozen and has a guard rejecting any forward call. Keypoint receives no loss; B's shared decoder changes can still affect keypoint predictions indirectly.

`install.py` adds one guarded `MAGROAD_INIT_ONLY` early return before original SAM initialization. It defaults false for all existing configs. The runner immediately loads the complete WildRoad state, allowing only new LoRA keys to be missing. It prints loaded/missing/unexpected keys, strips known repeated Lightning/module prefixes, rejects collisions and unrecognized keys, and explicitly excludes the old `mask_criterion.pos_weight` training-loss buffer when absent from the new loss. It never silently skips encoder, decoder, or topology weights. Real FP16 baseline equivalence against original mask inference was exact (max logit difference 0).

## Data preparation

Official directory:

```
LoveDA/Train/{Urban,Rural}/{images_png,masks_png}/*.png
LoveDA/Val/{Urban,Rural}/{images_png,masks_png}/*.png
```

Set `LOVEDA_ROOT`; names of split/domain are matched case-insensitively. Loader supports images/images_png and labels/masks/masks_png. Each image requires a matching semantic label. Duplicate IDs cause an error. For this remote installation only, Train RGB is separately stored in `processed/LoveDA/ssl_train_rgb/{urban,rural}`; specify `LOVEDA_TRAIN_RGB_ROOT`. This is an explicit alternate source, never guessed, and original files remain unchanged.

Official [LoveDA README](https://github.com/Junjue-Wang/LoveDA#dataset-and-contest): road=3, valid IDs=1..7, no-data=0. Loader yields RGB float32 HWC 0..255, road_mask float32 HW 0/1 and valid bool HW. Invalid values are excluded from BCE and Dice; sanity checker rejects unexpected IDs. No resize/crop. Conservative flips/rot90/brightness/contrast ±5%, config switch; never augment Val. `LOVEDA_DOMAIN=all|urban|rural`.

Train is 2522 pairs. Road pixels 134119839, non-road 2410756774, ignore 99632059; valid-road ratio 5.27019%. Raw imbalance suggests 17.9746, but default BCE pos_weight stays 1 to avoid uncontrolled false positives.

## Commands

Run from remote `/home/badger/sam-inference/MaGRoad`, using `/home/badger/overhead-ssl/.venv/bin/python`. Generic template configs contain empty user data/checkpoint paths; remote_* configs record this machine's paths.

```
python magroad_loveda/install.py
python magroad_loveda/tools/check_loveda_dataset.py --root /data/LoveDA --domain all
python magroad_loveda/tools/analyze_loveda_road.py --root /data/LoveDA --domain all
python -m pytest magroad_loveda/tests -q
python magroad_loveda/tools/smoke_test_loveda.py --config magroad_loveda/config/loveda/remote_encoder.yaml
python magroad_loveda/train.py --config magroad_loveda/config/loveda/remote_encoder.yaml --gradient-audit --max-steps 10 --limit-train 40 --limit-val 4 --output-dir magroad_loveda/outputs/smoke_10
python magroad_loveda/train.py --config magroad_loveda/config/loveda/remote_encoder.yaml --eval_target_domain --output-dir magroad_loveda/outputs/baseline
python magroad_loveda/train.py --config magroad_loveda/config/loveda/remote_encoder.yaml --eval-only --output-dir magroad_loveda/outputs/baseline
python magroad_loveda/train.py --config magroad_loveda/config/loveda/remote_encoder.yaml
# Only after A finishes and its validation/regression results are saved:
python magroad_loveda/train.py --config magroad_loveda/config/loveda/remote_encoder_decoder.yaml
python magroad_loveda/infer_loveda_adapter.py --config magroad_loveda/config/loveda/remote_encoder.yaml --checkpoint magroad_loveda/outputs/encoder/adapter_last.pt --input_dir /data/target_test --output_dir outputs/target_test
python magroad_loveda/train.py --config magroad_loveda/config/loveda/remote_encoder.yaml --resume magroad_loveda/outputs/encoder/checkpoints/last.ckpt
python magroad_loveda/tools/create_report.py
```

Sequential persistent runner: `python -u magroad_loveda/run_experiments.py`. It locks against duplicate runs and executes baseline -> A -> exact weight audit -> report -> B 10-step audit -> B -> report. Any subprocess failure stops the pipeline. It is not a scheduler and must not be restarted blindly over an existing completed run; resume interrupted training with `--resume`.

## Hardware, metrics and outputs

Defaults: native 1024, batch1, FP16, accumulation4, rank4, 15 epochs, AdamW .01, LoRA and decoder LR each 1e-4, Dice weight1. Optional early stopping patience4 on Val IoU. This LR matches the original encoder's reduced LR rather than decoder's original full 1e-3. No target/test threshold tuning.

Measured RTX 4070 Laptop 8GB: without checkpointing backward OOM (peak allocated 6877.5, reserved7028 MiB; extra384 MiB needed). Retain 1024; non-reentrant checkpointing of encoder blocks in only the training path succeeds (10 optimizer steps =40 microbatches: peak allocated3275.6, reserved4116 MiB). Input need not require gradients. Original SAM internals are untouched. `oom.json` and traceback preserve failures. Standard Lightning single-GPU/logger notices are not hidden. Unused inherited topology metric hooks are overridden instead of silencing their warnings.

`outputs/{baseline,encoder,encoder_decoder}/`: config/base SHA/parameter stats, training loss JSON, TensorBoard train/*, val/* and group lr; fixed Val panels per epoch; validation history with TP/FP/FN-derived aggregate IoU/Precision/Recall/F1 at t=.5 and threshold grid .1..9; threshold selected on Val F1; target_domain/baseline or epoch_NNN fixed 8 images with .5 and separate Val-best outputs. NPY stores exact float probabilities, PNG stores 16-bit probabilities, binary is0/255, panel has constant0..1 probability colors.

Full checkpoints `checkpoints/{best_road_iou,best_road_recall,last}.ckpt` include optimizer, scheduler, epoch, config/base reference and metric history for resume. `adapter_epoch_NNN.pt` and `adapter_last.pt` include LoRA only for A, LoRA+map_decoder for B, config and base checkpoint SHA. No optimizer in adapters. Full/adapter loading checks base SHA and experiment mode/key compatibility. Keep the base checkpoint available; never overwrite it.

Optional `--eval-only --resume FULL_CKPT` evaluates a trained full checkpoint. Adapter target inference accepts `--threshold-json path/to/road_threshold.json`. Always use the threshold metadata corresponding to the selected epoch, not blindly last epoch metadata.

## A* scope and decision gate

`planning/astar_eval.py`: probability, manually supplied integer xy start/goal, threshold and optional GT valid-path mask -> route_found, length, xy coordinates. Eight-connected search prohibits corner cutting. Non-autograd only. Tests continuous road, gap, alternative route and blocked low-probability shortcut. Human-reviewed GT pair schema is explicit in planning/graph_utils.py. No prediction-generated endpoints and no official target Route Success/Stretch/False Shortcut/APLS without reviewed GT.

After A/B, generate `outputs/LOVEDA_LORA_REPORT.md` with complete Val metrics, best IoU vs separately best Recall epoch, VRAM, time, trainable parameters and fixed before/after panels. No unlabelled XJTLU quantitative Recall claim, no single-seed statistical significance claim. Stop at report. Only if reviewed road predictions are mostly continuous should later centerline/topology/planning work be considered; otherwise investigate GSD/zoom, source/normalization, decoder and road label definition first.
