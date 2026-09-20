"""Summarize actual proposal audit and sensitivity without claiming route improvements."""
import json,csv,hashlib,html
from pathlib import Path
from collections import Counter
import numpy as np
from graph_utils import write_json


def main():
    root=Path('outputs/connection_candidates_v2/xjtlu_1');rows=json.loads((root/'all_candidates.json').read_text())['candidates'];s=json.loads((root/'candidate_statistics.json').read_text());coverage=json.loads((root/'pair_candidate_coverage_summary.json').read_text());active=[r for r in rows if r['status']=='proposed' and not r.get('budget_excluded')]
    for kind,d in s['types'].items():
        rr=[r for r in rows if r['type']==kind];d['budget_excluded']=sum(r['status']=='proposed' and bool(r.get('budget_excluded')) for r in rr);d['deduplicated_audit_rows']=sum(bool(r.get('deduplicated')) for r in rr)
        assert d['total']==d['hard_gate_rejected']+d['budget_excluded']+d['proposed']
    s['rejection_flag_counts']=dict(Counter(f for r in rows for f in r['rejection_flags']));write_json(root/'candidate_statistics.json',s)
    perps=[r for r in active if r['type']=='endpoint_edge' and r['geometry_features']['target_edge_angle_difference']>65];long=[r for r in active if r['type']=='component_corridor' and r['geometry_features']['straight_distance_px']>100]
    topstats={}
    for kind in s['types']:
        rr=sorted([r for r in active if r['type']==kind],key=lambda r:-r['heuristic_score'])[:50]
        topstats[kind]={'count':len(rr),'score_mean':float(np.mean([r['heuristic_score'] for r in rr])),'mean_probability':float(np.mean([r['probability_features']['mean'] for r in rr])),'mean_p10':float(np.mean([r['probability_features']['p10'] for r in rr])),'top_ids':[r['candidate_id'] for r in rr]}
    write_json(root/'top_candidate_statistics.json',topstats)
    sweep=list(csv.DictReader((root/'candidate_sweep.csv').open()));table='\n'.join(f"| {r['case']} | {r['candidate_count']} | {r['endpoint_edge']} | {r['endpoint_junction']} | {r['corridors']} | {float(r['runtime_seconds']):.1f} | {r['coverage_failures_any_failed_endpoint']}/18 | {r['connectivity_direct_pair']}/14 |" for r in sweep)
    counts='\n'.join(f"| {k} | {d['total']} | {d['hard_gate_rejected']} | {d['budget_excluded']} | {d['proposed']} |" for k,d in s['types'].items());top='\n'.join(f"| {k} | {v['count']} | {v['score_mean']:.3f} | {v['mean_probability']:.3f} | {v['mean_p10']:.3f} |" for k,v in topstats.items())
    report=f'''# XJTLU Connection Candidate Generator V2 实验报告

结论：**B。仍有大量失败区域没有直接候选覆盖，需要继续改 proposal mechanism 或回到前端 perception。** 本阶段只产出候选，未接受连接、未修改 raw/rule graph、未训练任何模型。原正式 A* baseline 仍为 9/41；本阶段没有运行接受候选后的 A*。

## 1. 新增 / 修改文件

新增 `scripts/candidate_generator_v2.py`、`scripts/corridor_proposal.py`、`scripts/visualize_connection_candidates.py`、`scripts/evaluate_candidate_coverage.py`、`scripts/run_candidate_sweep.py`、`scripts/create_candidate_synthetic_examples.py`、`scripts/report_connection_candidates_v2.py`；几何、局部生成和加速内核位于 `scripts/connection_candidates/`。新增 `tests/test_connection_candidates_v2.py` 和 `docs/connection_candidates_v2.md`。现有 MaGRoad、graph parser、rule refinement 和模型参数未改写。

## 2–10. 数量与拒绝审计

输入 raw graph：2494 nodes / 2363 edges / 229 components；444 endpoints、274 junctions。局部半径80 px，回看30 px，朝向门限45°，junction snap8 px。相同 component 的局部候选允许保留。

| 类型 | 审计总数 | hard-gate 拒绝 | 排序预算/重复排除 | 保留 proposed |
|---|---:|---:|---:|---:|
{counts}

保留合计 **{len(active)}**；hard-gate 拒绝合计 **{sum(d['hard_gate_rejected'] for d in s['types'].values())}**。总数含多个搜索的分段子候选和被拒绝原路径，不等于独立连接数量。JSON 同时保存硬过滤和预算标志；统计表按互斥状态计数。

空间索引筛出 {s['component_pairs']} 个邻近 component pair；执行 {s['pixel_searches']} 次局部 pixel A*。默认每个实际 component pair 最多保留3个 corridor，每个 endpoint 最多5个局部候选。邻居上限10应用于每个 component 的主动选择；无向并集的最终度数可能超过10。

## 11. 性能

默认 endpoint 生成 **{s['endpoint_runtime_seconds']:.2f}s**，corridor 阶段 **{s['corridor_runtime_seconds']:.2f}s**，总生成 **{s['total_runtime_seconds']:.2f}s**。不含 JSON 写出、绘图、pair coverage 和 sweep。平均 search window **{s['mean_window_pixels']:.0f} pixels**，单次搜索中位数 **{s['median_search_seconds']*1000:.2f}ms**。局部窗口64 px margin，seed disk3 px；原图概率图11,554,816 pixels，窗口远小于全图。代价图只构建一次。

八邻域代价为 `step_distance * (cost[u]+cost[v])/2`，`cost=1+8*(1-p)^2+4*(p<0.2)`。有限低概率代价允许树荫缺口搜索。C++ 内核通过与 Python 参考实现的最优路径代价一致性测试；默认运行已使用缓存编译内核。

## 12. Top candidate score / probability

| 类型 | top数量（最多50） | score均值 | road prob均值 | p10均值 |
|---|---:|---:|---:|---:|
{top}

所有 heuristic 子分数、概率统计、方向、长度、曲率和 graph crossing 都保留。分数仅用于审核排序，无 `label` 字段。PNG 是 fused prediction 的 uint8/255，量化步长1/255；概率不是语义真值。

## 13–15. 41 pair coverage

41有效pair中9个baseline成功、18个projection coverage failure、14个connectivity failure。pair_35 继续排除，未改坐标或投影40 px限制。

- 18 projection failures：**{coverage['coverage_failures_with_candidate_near_at_least_one_failed_endpoint']}/18** 在至少一个失败端点40 px内存在保留候选；**{coverage['coverage_failures_with_candidates_near_all_failed_endpoints']}/18** 在所有失败端点附近都有候选。
- 14 connectivity failures：**{coverage['connectivity_failures_with_direct_component_pair_proposal']}/14** 存在连接两个最近投影component的直接候选；**{coverage['connectivity_failures_with_adjacent_component_proposal']}/14** 至少存在涉及其中一个component的 edge/junction/corridor 候选。
- 邻接候选覆盖是较弱诊断，不能代替直接 component pair 覆盖。以上仅表示候选几何存在，不证明 accept 后能找到正确道路，不证明 41 pair success 提升。

详见 `pair_candidate_coverage.json`；18个投影失败的 start/goal 诊断卡位于 `coverage_failure_review/`。点击点仅诊断，未进入proposal seed或训练数据。

## 16. 视觉合理的 T-junction proposal

`candidate_000019`：endpoint-edge，距离21.2 px，mean prob0.604、p10 0.525。RGB 中可见水平道路与支路交汇附近，投影提供已有edge中间的attachment，支持T连接审核。它是视觉合理示例，尚无人审GT。

[查看审核卡](selected_reviews/t_pattern_candidate_000019.png)。几何近垂直的edge候选共 **{len(perps)}**；608个edge候选来自223个不同endpoint，远多于43个endpoint-endpoint。不能把50个几何模式计为50个已恢复真实T路口。

## 17. 明显可疑的 endpoint-edge proposal

`candidate_003197`：距离72.9 px，mean0.087、p10 0，斜穿树冠/建筑间区域，RGB缺乏连续道路支持。几何朝向通过但概率排名很低，保留它是为了后续 reject/hard-negative 审核。局部候选阶段没有把最低概率作为统一硬门限。

[查看审核卡](selected_reviews/low_probability_edge_candidate_003197.png)。跨机动车道的短横向候选（如000043）也不能直接视为步行可通行连接。

## 18. 视觉合理的长 corridor proposal

`candidate_007480`：跨度116.7 px，路径125.8 px，detour1.08，mean0.714、p10 0.606。路径沿可见主干道路接入路口，而非直接跨背景画直线。`candidate_006382` 亦沿同一主路走廊，仍需核实是否属于重复道路中心线。

[长corridor审核卡](selected_reviews/long_corridor_candidate_007480.png)。保留corridor中 **{len(long)}** 个跨度>100 px。

## 19. 可疑 / 无法确认的 corridor proposal

`candidate_010517`：跨度93.0 px、mean0.359、p10 0.316，但RGB中沿建筑阴影/边界附近横穿，概率响应不能证明道路通行。`candidate_009836` 穿过大段树冠，RGB也不足以确认。这些是必须人工审查的潜在错误，不能在缺GT时断言语义误标。

[阴影边界可疑卡](selected_reviews/campus_candidate_010517.png)、[树冠可疑卡](selected_reviews/campus_candidate_009836.png)。third-component crossing 会拒绝无视中途拓扑的原corridor，并生成明确分段子候选；进一步不明确的分段仍拒绝。

## 20. 七组单因素敏感性检查

默认80/250/250，加edge radius40/120、component radius150/350、max length150/350；不是27组合grid，未根据test A* success选参数。

| 变化 | 全部保留 | edge | junction | corridor | 生成秒 | failed端点附近 | 直接component pair |
|---|---:|---:|---:|---:|---:|---:|---:|
{table}

详见 `candidate_sweep.csv`。候选增加不能直接计为道路召回提升；半径/预算敏感性同时受closest seed预选和ranking影响。

## 21. 是否适合下一阶段 classifier

当前规模可管理，可供人工候选审核与GT matching；**尚不能直接训练**，仍缺人审完整centerline GT以及accept/reject匹配。默认共有{len(active)}候选，保存拒绝项可用于检查漏召回。需特别增加建筑边、平行机动车道横连、树冠、河岸、操场等hard-negative审核。没有任何自动接受的连接。

## 22. 未覆盖类型与下一步判断

至少11/18 projection failures的失败端点附近没有候选，9/14 connectivity failures没有直接component-pair候选。仅有组件间搜索不能从完全缺失道路中产生独立graph节点；长于搜索半径/长度门限的整段缺失道路、低概率道路、非endpoint局部附着和seed预算未选中的连接仍可能遗漏。扩大空间参数未必解决前端概率缺失或错误语义。

Q1：endpoint-edge提供明确额外attachment模式和50个近垂直几何候选，但无GT不能量化真实T召回。Q2：64个junction候选保留明确共享junction id和empty-sector特征；靠近junction的edge投影已去重转换，不能把两类计为独立召回相加。Q3：可见长corridor有合理示例，同时存在阴影/树冠可疑例。Q4：保守直接覆盖为7/18失败端点附近、5/14component pairs。Q5：候选规模可控，足以审核，不能据此保证classifier训练有效。

**最终选择B：继续完善 proposal / perception，并先人工审核。停止于candidate generation，不自动训练classifier。**

验证：13项新单元测试加11项既有测试，共24 passed；五种synthetic maps位于 `../synthetic/`，包含straight、parallel、T、obstacle、blank。原graph和模型只读，provenance记录输入/代码SHA。无接受后的graph，无新的navigation-ready或A*提升结论。
'''
    (root/'final_report.md').write_text(report)
    items=[]
    for kind,data in topstats.items():
        folder='corridor_reviews' if kind=='component_corridor' else kind+'_reviews'
        items.append(f'<h2>{kind}</h2>')
        for identifier in data['top_ids']:items.append(f'<p>{identifier}</p><img loading="lazy" style="max-width:100%" src="{folder}/{identifier}.png">')
    (root/'review_index.html').write_text('<!doctype html><meta charset="utf-8"><title>XJTLU candidates V2</title><body style="background:#151515;color:white;font-family:sans-serif"><h1>Candidate proposals only</h1><p>Green: source component; yellow: target. Magenta: candidate. No accepted roads or GT labels.</p>'+''.join(items))
    print('report written; sweep cases',len(sweep))
if __name__=='__main__':main()
