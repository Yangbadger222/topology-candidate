from dataclasses import dataclass, field
import math
from .graph_io import RoadGraph

@dataclass
class Candidate:
    sample_id: str; scene_id: str; source_component: int; source_node: int; target_component: int
    target_x: float; target_y: float; gap_px: float; source_category: str
    features: dict[str, float] = field(default_factory=dict)
    def to_dict(self):
        return {"sample_id": self.sample_id, "scene_id": self.scene_id, "source_component": self.source_component,
                "source_node": self.source_node, "target_component": self.target_component, "target_x": self.target_x,
                "target_y": self.target_y, "gap_px": self.gap_px, "source_category": self.source_category,
                "features": self.features}

def _segment_projection(px, py, ax, ay, bx, by):
    dx,dy=bx-ax,by-ay; den=dx*dx+dy*dy; t=0 if den==0 else max(0,min(1,((px-ax)*dx+(py-ay)*dy)/den))
    return ax+t*dx, ay+t*dy

def generate_candidates(graph: RoadGraph, scene_id="scene", radius=128.0, top_k=3, main_component=None, support=None, heat_support=None):
    if main_component is None: main_component=max(graph.components,key=lambda cid: len(graph.components[cid]),default=None)
    rows=[]
    for cid, comp in graph.components.items():
        if cid == main_component: continue
        for node_id in graph.endpoints(comp):
            p=graph.nodes[node_id]; targets=[]
            for tid,tcomp in graph.components.items():
                if tid == cid: continue
                for a,b in graph.edges:
                    if a not in tcomp or b not in tcomp: continue
                    qx,qy=_segment_projection(p.x,p.y,graph.nodes[a].x,graph.nodes[a].y,graph.nodes[b].x,graph.nodes[b].y)
                    d=math.hypot(p.x-qx,p.y-qy)
                    if d <= radius: targets.append((d,tid,qx,qy))
            # endpoint-to-endpoint fallback covers isolated/short graphs without edges
            for tid,tcomp in graph.components.items():
                if tid == cid: continue
                for other in graph.endpoints(tcomp):
                    q=graph.nodes[other]; d=math.hypot(p.x-q.x,p.y-q.y)
                    if d <= radius: targets.append((d,tid,q.x,q.y))
            best={ (t[1], round(t[2],4), round(t[3],4)):t for t in sorted(targets) }
            for rank,(d,tid,x,y) in enumerate(sorted(best.values())[:top_k]):
                fs={"gap_distance":d,"normalized_gap":d/max(radius,1),"component_nodes":len(comp),
                    "component_length_px":graph.component_length(comp),"source_degree":graph.graph.degree[node_id],
                    "target_component_nodes":len(graph.components[tid]),"target_is_main":float(tid==main_component),
                    "gap_graph_support":float(support(p.x,p.y,x,y)) if callable(support) else 0.0,
                    "gap_heat_support":float(heat_support(p.x,p.y,x,y)) if callable(heat_support) else 0.0}
                rows.append(Candidate(f"{scene_id}_C{cid}_to_C{tid}_{node_id:03d}_{rank:03d}",scene_id,cid,node_id,tid,x,y,d,
                    "isolated" if len(comp)==1 else "floating_short",fs))
    return rows
