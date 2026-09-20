"""Local multi-source eight-connected pixel A*: proposals, not graph edges."""
import heapq,math,time
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Point,LineString,box
from shapely.strtree import STRtree
from shapely.ops import substring
from graph_utils import xy
from connection_candidates.geometry import make_candidate,angle


def python_pixel_astar(cost,source,target,radius=3,origin=(0,0)):
    h,w=cost.shape;source=np.asarray(source)-origin;target=np.asarray(target)-origin
    def disk(q):
        return [(x,y) for y in range(max(0,math.floor(q[1]-radius)),min(h,math.ceil(q[1]+radius)+1)) for x in range(max(0,math.floor(q[0]-radius)),min(w,math.ceil(q[0]+radius)+1)) if np.linalg.norm(np.array([x,y])-q)<=radius]
    starts=disk(source);goals=set(disk(target))
    if not starts or not goals:return None,0
    dist=np.full((h,w),np.inf);parent={};queue=[];expanded=0
    def heuristic(x,y):return max(0,math.hypot(x-target[0],y-target[1])-radius)
    for x,y in starts:dist[y,x]=0;heapq.heappush(queue,(heuristic(x,y),0,x,y))
    steps=[(dx,dy,math.hypot(dx,dy)) for dy in (-1,0,1) for dx in (-1,0,1) if dx or dy]
    while queue:
        _,d,x,y=heapq.heappop(queue)
        if d>dist[y,x]:continue
        expanded+=1
        if (x,y) in goals:
            path=[(x,y)]
            while path[-1] in parent:path.append(parent[path[-1]])
            return (np.asarray(path[::-1])+origin).tolist(),expanded
        for dx,dy,step in steps:
            xx,yy=x+dx,y+dy
            if not (0<=xx<w and 0<=yy<h):continue
            nd=d+step*(float(cost[y,x])+float(cost[yy,xx]))*.5
            if nd<dist[yy,xx]:dist[yy,xx]=nd;parent[xx,yy]=(x,y);heapq.heappush(queue,(nd+heuristic(xx,yy),nd,xx,yy))
    return None,expanded


def pixel_astar(cost,source,target,radius=3,origin=(0,0)):
    if not np.isfinite(cost).all() or np.min(cost)<1:raise ValueError('A* cost must be finite and >=1')
    from connection_candidates.native_search import search
    try:return search(cost,source,target,radius,origin)
    except (OSError, __import__('subprocess').CalledProcessError):return python_pixel_astar(cost,source,target,radius,origin)


def component_pairs(ctx,c):
    boxes=[box(*z['bbox_xyxy']) for z in ctx.component_info];tree=STRtree(boxes);pairs=set()
    for i,b in enumerate(boxes):
        hits=[(b.distance(boxes[int(j)]),int(j)) for j in tree.query(b.buffer(c.component_search_radius)) if int(j)!=i and b.distance(boxes[int(j)])<=c.component_search_radius]
        for _,j in sorted(hits)[:c.max_component_neighbors]:pairs.add(tuple(sorted((i,j))))
    return sorted(pairs)


def seeds(ctx,i,j):
    g=ctx.g
    def nodes(k):
        info=ctx.component_info[k];return info['endpoints'] or info['boundary_nodes']
    aa,bb=nodes(i),nodes(j);rows=[]
    # Only spatially selected component pairs; seed pairs are locally ranked.
    for a in aa:
        for b in bb:
            rows.append((np.linalg.norm(xy(g,a)-xy(g,b)),{'node_id':a,'xy':xy(g,a).tolist(),'seed_type':'endpoint' if g.degree(a)==1 else ('junction' if g.degree(a)>=3 else 'edge'),'component_id':i},{'node_id':b,'xy':xy(g,b).tolist(),'seed_type':'endpoint' if g.degree(b)==1 else ('junction' if g.degree(b)>=3 else 'edge'),'component_id':j}))
    # Exact projections also permit endpoint-edge / edge-edge attachment.
    for k,l in ((i,j),(j,i)):
        target_edges=[e for e,(u,v) in enumerate(ctx.edges) if ctx.component[u]==l]
        for n in nodes(k):
            if not target_edges:continue
            pt=Point(xy(g,n));e=min(target_edges,key=lambda e:ctx.lines[e].distance(pt));line=ctx.lines[e];t=line.project(pt);q=list(line.interpolate(t).coords[0]);source={'node_id':n,'xy':xy(g,n).tolist(),'seed_type':'endpoint' if g.degree(n)==1 else 'edge','component_id':k};target={'edge_id':e,'edge_nodes':list(ctx.edges[e]),'projection_t':t/max(line.length,1e-9),'xy':q,'seed_type':'edge','component_id':l}
            rows.append((line.distance(pt),source,target))
    unique={}
    for d,a,b in sorted(rows,key=lambda z:z[0]):
        key=tuple(np.round(np.concatenate([a['xy'],b['xy']]),1))
        if key not in unique:unique[key]=(d,a,b)
    return list(unique.values())


def generate_corridors(ctx,c):
    # Geometry-only mode uses a neutral raster solely for path-cost calculation;
    # output probability_features retain evidence_available=False.
    p=ctx.p if ctx.p is not None else np.full((max(2,int(np.ceil(max((n[1] for n in ctx.g.nodes(data='y')),default=1)))+2), max(2,int(np.ceil(max((n[1] for n in ctx.g.nodes(data='x')),default=1)))+2)), .5, dtype=np.float32)
    cost=(1+c.corridor_lambda_prob*(1-p)**c.corridor_gamma+c.low_probability_penalty*(p<c.low_prob_threshold)).astype(np.float32)
    pairs=component_pairs(ctx,c);rows=[];logs=[]
    for pair_number,(i,j) in enumerate(pairs):
        if pair_number%100==0:print(f"corridor pair {pair_number}/{len(pairs)} searches={len(logs)}",flush=True)
        proposals=[]
        for distance,a,b in seeds(ctx,i,j)[:c.max_corridors_per_component_pair*2]:
            if distance>c.component_search_radius or distance<2:continue
            q=np.asarray([a['xy'],b['xy']]);lo=np.maximum(0,np.floor(q.min(0)-c.corridor_window_margin)).astype(int);hi=np.minimum([p.shape[1],p.shape[0]],np.ceil(q.max(0)+c.corridor_window_margin+1)).astype(int)
            crop=cost[lo[1]:hi[1],lo[0]:hi[0]];start=time.perf_counter();raw,expanded=pixel_astar(crop,a['xy'],b['xy'],c.seed_radius,lo)
            logs.append({'components':[i,j],'window_xyxy':[*lo.tolist(),*hi.tolist()],'window_pixels':crop.size,'seconds':time.perf_counter()-start,'expanded':expanded,'found':raw is not None})
            if raw is None:continue
            raw=[a['xy']]+raw+[b['xy']];line=LineString(raw);simple=list(map(list,line.simplify(c.corridor_simplify_epsilon).coords));q=np.asarray(raw);vectors=np.diff(q,axis=0);vectors=vectors[np.linalg.norm(vectors,axis=1)>1e-8];turns=[angle(x,y) for x,y in zip(vectors,vectors[1:])]
            row=make_candidate(ctx,'component_corridor',a,b,raw,{'mean_turning_angle':float(np.mean(turns)) if turns else 0,'max_turning_angle':max(turns,default=0),'curvature':sum(np.radians(turns))/max(line.length,1),'existing_component_distance':box(*ctx.component_info[i]['bbox_xyxy']).distance(box(*ctx.component_info[j]['bbox_xyxy']))})
            row['geometry'].update(raw_pixel_path=raw,simplified_polyline=simple,simplification_hausdorff_px=line.hausdorff_distance(LineString(simple)))
            flags=[]
            if line.length>c.max_corridor_length:flags.append('max_corridor_length')
            if row['geometry_features']['detour_ratio']>c.max_detour_ratio:flags.append('max_detour_ratio')
            if row['probability_features']['mean']<c.min_mean_probability:flags.append('min_mean_probability')
            if row['probability_features']['p10']<c.min_p10_probability:flags.append('min_p10_probability')
            third=[z for z in row['graph_features']['crossing_locations'] if z['component_id'] not in (i,j)]
            # Do not emit a connector ignoring an intermediate component.
            if third:
                flags.append('third_component_requires_split');row['split_proposals']=[{'component_id':z['component_id'],'xy':z['xy'],'path_position':z['path_position']} for z in sorted(third,key=lambda z:z['path_position'])]
            row['rejection_flags']=flags;row['status']='rejected_by_hard_gate' if flags else 'proposed';proposals.append(row)
            if third:
                stops=[(0,a)]
                for hit in sorted(third,key=lambda z:z['path_position']):
                    if hit['component_id']==stops[-1][1]['component_id']:continue
                    stops.append((hit['path_position'],{'component_id':hit['component_id'],'edge_id':hit['edge_id'],'edge_nodes':list(ctx.edges[hit['edge_id']]),'xy':hit['xy'],'seed_type':'edge'}))
                stops.append((line.length,b))
                for (sa,aa),(sb,bb) in zip(stops,stops[1:]):
                    if aa['component_id']==bb['component_id'] or sb-sa<2:continue
                    segment=list(map(list,substring(line,sa,sb).coords));child=make_candidate(ctx,'component_corridor',aa,bb,segment,{'split_from_component_pair':[i,j],'split_path_interval':[sa,sb]})
                    child['geometry'].update(raw_pixel_path=segment,simplified_polyline=list(map(list,LineString(segment).simplify(c.corridor_simplify_epsilon).coords)))
                    child_flags=[]
                    if child['geometry']['length_px']>c.max_corridor_length:child_flags.append('max_corridor_length')
                    if child['geometry_features']['detour_ratio']>c.max_detour_ratio:child_flags.append('max_detour_ratio')
                    if child['probability_features']['mean']<c.min_mean_probability:child_flags.append('min_mean_probability')
                    if child['probability_features']['p10']<c.min_p10_probability:child_flags.append('min_p10_probability')
                    if any(z['component_id'] not in (aa['component_id'],bb['component_id']) for z in child['graph_features']['crossing_locations']):child_flags.append('third_component_requires_split')
                    child['rejection_flags']=child_flags;child['status']='rejected_by_hard_gate' if child_flags else 'proposed';proposals.append(child)
        proposed=0
        for row in sorted(proposals,key=lambda r:-r['heuristic_score']):
            row['budget_excluded']=row['status']=='proposed' and proposed>=c.max_corridors_per_component_pair
            if row['status']=='proposed' and not row['budget_excluded']:proposed+=1
            rows.append(row)
    dedup={};budget={}
    for row in sorted(rows,key=lambda r:-r['heuristic_score']):
        pair=tuple(sorted([row['source']['component_id'],row['target']['component_id']]))
        aa,bb=row['source']['xy'],row['target']['xy'];aa,bb=(aa,bb) if row['source']['component_id']<row['target']['component_id'] else (bb,aa);key=(pair,tuple(np.round(aa,1)),tuple(np.round(bb,1)))
        if key in dedup:row['budget_excluded']=True;row['deduplicated']=True
        else:
            dedup[key]=True
            if row['status']=='proposed':
                row['budget_excluded']=budget.get(pair,0)>=c.max_corridors_per_component_pair
                if not row['budget_excluded']:budget[pair]=budget.get(pair,0)+1
    return rows,{'component_pairs':len(pairs),'pixel_searches':len(logs),'mean_window_pixels':float(np.mean([r['window_pixels'] for r in logs])) if logs else 0,'median_search_seconds':float(np.median([r['seconds'] for r in logs])) if logs else 0,'search_log':logs}
