import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Point
from graph_utils import xy
from .geometry import angle,make_candidate,oriented


def generate_local(ctx,c):
    g=ctx.g;rows={};audit={'spatial_hits':0,'deduplicated':0,'budget_excluded':0}
    def node(n):return {'node_id':n,'seed_type':'endpoint' if g.degree(n)==1 else 'junction','xy':xy(g,n).tolist(),'component_id':ctx.component[n]}
    def add(u,target,kind,extra):
        v=target.get('node_id');key=(kind,min(u,v),max(u,v)) if kind=='endpoint_endpoint' else (kind,u,target.get('edge_id',v))
        if key in rows:audit['deduplicated']+=1;return
        delta=np.asarray(target['xy'])-xy(g,u);distance=np.linalg.norm(delta)
        if distance<1e-4:return
        aa=angle(ctx.tangents[u],delta);angles=[aa]
        if kind=='endpoint_endpoint':angles.append(angle(ctx.tangents[v],-delta))
        flags=[]
        if max(angles)>c.endpoint_edge_angle:flags.append('outward_tangent_angle')
        features={'euclidean_distance':float(distance),'outward_tangent_A':ctx.tangents[u].tolist(),'connection_direction':(delta/distance).tolist(),'max_angle_error':max(angles),'angle_A':aa,**extra}
        if kind=='endpoint_endpoint':features.update(outward_tangent_B=ctx.tangents[v].tolist(),angle_B=angles[1])
        row=make_candidate(ctx,kind,node(u),target,[xy(g,u),target['xy']],features,flags)
        if any(z['kind']=='interior' for z in row['graph_features']['crossing_locations']):row['rejection_flags'].append('local_connector_interior_crossing');row['status']='rejected_by_hard_gate'
        rows[key]=row
    if len(ctx.ends)>1:
        for i,j in sorted(cKDTree([xy(g,n) for n in ctx.ends]).query_pairs(c.endpoint_endpoint_radius)):
            audit['spatial_hits']+=1;u,v=ctx.ends[i],ctx.ends[j]
            if not g.has_edge(u,v):add(u,node(v),'endpoint_endpoint',{})
    jt=cKDTree([xy(g,n) for n in ctx.junctions]) if ctx.junctions else None
    def junction(u,v):
        dirs=[oriented(g,v,w)[1]-xy(g,v) for w in g[v]];direction=xy(g,u)-xy(g,v);separation=min(angle(direction,d) for d in dirs)
        add(u,node(v),'endpoint_junction',{'junction_degree':g.degree(v),'junction_incident_directions':[d.tolist() for d in dirs],'junction_direction_compatibility':min(1,separation/90),'empty_sector_min_angle':separation})
    for u in ctx.ends:
        if jt:
            for j in sorted(jt.query_ball_point(xy(g,u),c.endpoint_edge_radius)):
                v=ctx.junctions[j];audit['spatial_hits']+=1
                if not g.has_edge(u,v):junction(u,v)
        for i in sorted(map(int,ctx.tree.query(Point(xy(g,u)).buffer(c.endpoint_edge_radius)))):
            a,b=ctx.edges[i]
            if u in (a,b):continue
            line=ctx.lines[i];point=Point(xy(g,u));distance=line.distance(point)
            if distance>c.endpoint_edge_radius:continue
            audit['spatial_hits']+=1;t=line.project(point);q=np.asarray(line.interpolate(t).coords[0]);near=min((a,b),key=lambda n:np.linalg.norm(xy(g,n)-q))
            if np.linalg.norm(xy(g,near)-q)<=c.junction_snap_radius:
                if g.degree(near)>=3:junction(u,near);continue
                if g.degree(near)==1:
                    if np.linalg.norm(xy(g,near)-xy(g,u))<=c.endpoint_endpoint_radius:add(u,node(near),'endpoint_endpoint',{})
                    continue
            # Degree-two edge boundaries remain valid attachment points.
            eps=min(2,line.length/4);direction=np.asarray(line.interpolate(min(line.length,t+eps)).coords[0])-line.interpolate(max(0,t-eps)).coords[0];err=min(angle(q-xy(g,u),direction),angle(q-xy(g,u),-direction))
            add(u,{'edge_id':i,'edge_nodes':[a,b],'projection_t':t/line.length,'projection_distance_px':t,'xy':q.tolist(),'seed_type':'edge','component_id':ctx.component[a]},'endpoint_edge',{'target_edge_angle_difference':err,'connection_pattern':'near_parallel' if err<25 else ('near_perpendicular' if err>65 else 'oblique')})
    ordered=sorted(rows.values(),key=lambda r:(-r['heuristic_score'],r['type'],str(r['source']),str(r['target'])))
    counts={};retained=[]
    for row in ordered:
        ends=[row['source']['node_id']]+([row['target']['node_id']] if row['type']=='endpoint_endpoint' else [])
        if row['status']=='proposed' and any(counts.get(n,0)>=c.max_candidates_per_endpoint for n in ends):
            row['budget_excluded']=True;audit['budget_excluded']+=1
        else:
            row['budget_excluded']=False
            if row['status']=='proposed':
                for n in ends:counts[n]=counts.get(n,0)+1
        retained.append(row)
    return retained,audit
