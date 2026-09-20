# 人工 navigation centerline GT 标注规范 v1

## 可通行定义
v1 默认 pedestrian navigation：包含公开可通行机动车路、校园内部路、人行道、步行小路和服务道路；不能混同车辆导航。车辆实验必须按 path_type 过滤并另建 GT/splits。未确认能否通行标 uncertain，在提交 coverage_complete 之前人工复核，不凭阴影或模型预测判断。

排除楼顶/建筑边界、树边、阴影边、河岸、车位线、运动场跑道及不可通行广场边缘。能依据道路两端和独立影像确认的树遮挡道路可以画连续中心线，同时记录 visibility=occluded、verification_source。无法确认的连接不补。

## 如何画
画中心线 graph，不画 mask/polygon。弯道增加折点；交叉可通行处共享同一 node id；仅线段交叉不代表连通，须显式拆分为 junction。立交/不同层不连；尽可能绘制实际道路交汇位置而不是中心点直连穿建筑。平行独立人行道/机动车路分别画，仅在可通行的横穿点连接。

每个标注区域必须彻底检查所有路。空区域只有确认确实无路时才能 coverage_complete=true，否则缺标不能作 negative。GT禁止用MaGRoad输出自动生成。D原始人工片段可重用，预测图仅供后续错误比对，第一轮请用纯RGB标注。

## 坐标与 JSON
所有本阶段数据统一 x,y、左上角原点、units=inference_pixel；image_size=[W,H]。XJTLU推理尺寸4096x2821；从原图标注转换必须显式分别乘4096/8192与2821/5642。D图保持1024x1024，不为了尺寸一致自动缩放。各自cache必须与标注图同坐标。

```
{
  "coordinate_order":"x,y", "units":"inference_pixel",
  "image_size":[4096,2821], "region_id":"xjtlu_1",
  "annotation_status":"reviewed", "coverage_complete":true,
  "navigation_profile":"pedestrian",
  "nodes":[{"id":0,"x":100,"y":200},{"id":1,"x":150,"y":200}],
  "edges":[{"u":0,"v":1,"polyline":[[100,200],[150,200]],
             "path_type":"pedestrian_path","visibility":"visible"}]
}
```
上面仅格式示例，不是可用训练GT。工具导出draft、coverage_complete=false；人工检查坐标、完整性、junction和path_type后才改reviewed/true。双人复核至少validation/test全部、train抽查20%。每次审核保存reviewer/date及版本。

## 需要多少数据
先审核已有D的76张1024图、四region、2788片段；再完整标注XJTLU独立test区域约20–30张1024等效图块，并在全图边界核对跨patch连接。建议第一轮总80–120张等效patch，数量不是统计充分性的保证。按标签生成结果调整，争取train至少500正+500负 existing edge、200正+400负 gap；validation/test各至少100正edge和50正gap，有足够建筑/河岸/阴影hard negatives。不足就补独立地理区域，不能挪test进train。

## split 与 A*
建议train=tiger_hill+wuzhong_wanda，validation=tongli_old_town，test=XJTLU独立完整区域；shihu_scenic仅2图暂不用于正式test。该分配是提案，须核实地理范围、相邻上下文隔离和样本数量。同一XJTLU大图所有相邻图块统一test，禁止按512随机split。split中bbox是包含encoder上下文的完整范围，跨split同一geographic_frame至少隔512推理像素。

在test区点击20–50真实start/goal，覆盖主干道、校园内部、树荫、跨block和多路口长距离。标注category和人工确认的expected_reachable，不得挑选模型已成功路线。保留本应不连通的负对。graph和pairs工具均输出推理坐标；正式eval只能消费test pairs。

## 命令
```
python3 scripts/create_astar_pairs.py --image /Users/badger/Desktop/xjtlu_1.png --image-size 4096 2821 --region xjtlu_1 --split test --mode graph --output outputs/learned_graph_refiner_v1/annotation_tools/xjtlu_graph.html
# --mode pairs 生成起终点工具
python3 scripts/build_learned_refiner_dataset.py --splits reviewed_splits.json --labels-only --output labeled_dataset
# 缓存后移除--labels-only并指定--magroad-root和每个region的cache/image_sha256
```
数据入口要求每个region字段：id、split、geographic_frame、bbox_xyxy、units、prediction_graph、gt、cache、image_sha256。绝对路径，明确声明原图SHA256。顶层min_cross_split_separation=512、regions=[...]。当前splits.json为空且status=awaiting_reviewed_gt，是阻止训练的状态文件，不是完成了split的声明。
