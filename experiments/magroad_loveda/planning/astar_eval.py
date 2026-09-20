"""Non-differentiable pixel A*, evaluation only. Coordinates are (x,y).
No prediction-derived endpoints, graph extraction, training loss, or success claims.
"""
import heapq,math
import numpy as np


def astar_eval(probability,start_xy,goal_xy,threshold=.5,gt_valid_path=None):
    p=np.asarray(probability)
    if p.ndim!=2 or not np.isfinite(p).all(): raise ValueError('Finite 2D probability required')
    if not 0<=threshold<=1 or np.any((p<0)|(p>1)): raise ValueError('Probability/threshold must be 0..1')
    allowed=p>=threshold
    if gt_valid_path is not None:
        gt=np.asarray(gt_valid_path,dtype=bool)
        if gt.shape!=p.shape: raise ValueError('GT valid-path shape mismatch')
        allowed &= gt
    def point(xy):
        if len(xy)!=2 or any(int(v)!=v for v in xy): raise ValueError('Integer pixel xy required; no automatic endpoint projection')
        return tuple(map(int,xy))
    start,goal=point(start_xy),point(goal_xy); h,w=p.shape
    fail={'route_found':False,'route_length':None,'route_coordinates':[]}
    if any(not (0<=x<w and 0<=y<h) for x,y in (start,goal)): return {**fail,'reason':'endpoint_out_of_bounds'}
    if not allowed[start[1],start[0]] or not allowed[goal[1],goal[0]]: return {**fail,'reason':'endpoint_blocked'}
    queue=[(math.dist(start,goal),0.,start)]; distance={start:0.}; parent={}
    while queue:
        _,g,xy=heapq.heappop(queue)
        if g!=distance[xy]: continue
        if xy==goal:
            path=[goal]
            while path[-1]!=start: path.append(parent[path[-1]])
            return {'route_found':True,'route_length':g,'route_coordinates':[list(x) for x in reversed(path)]}
        x,y=xy
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
            nx,ny=x+dx,y+dy
            if not(0<=nx<w and 0<=ny<h) or not allowed[ny,nx]: continue
            # Do not cut a diagonal across a blocked corner.
            if dx and dy and (not allowed[y,nx] or not allowed[ny,x]): continue
            ng=g+math.hypot(dx,dy); nxt=(nx,ny)
            if ng<distance.get(nxt,float('inf')):
                distance[nxt]=ng; parent[nxt]=xy; heapq.heappush(queue,(ng+math.dist(nxt,goal),ng,nxt))
    return {**fail,'reason':'disconnected'}
