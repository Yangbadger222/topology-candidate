"""Geometry/topology candidate proposals; MaGRoad evidence is optional."""
from dataclasses import dataclass, field
import math
from .graph_io import RoadGraph, Edge
from .source_mining import SourceSubgraph, mine_sources

@dataclass
class Candidate:
    sample_id: str
    scene_id: str
    source_component: int
    source_node: int
    target_component: int
    target_x: float
    target_y: float
    gap_px: float
    source_category: str
    features: dict[str,float] = field(default_factory=dict)
    source_subgraph_id: str = ""
    target_type: str = "edge"
    target_edge_id: str | None = None
    target_node_id: int | None = None
    rank: int = 1

    @property
    def candidate_id(self): return self.sample_id
    @property
    def source_endpoint_id(self): return self.source_node

    def to_dict(self):
        return {"candidate_id":self.sample_id,"sample_id":self.sample_id,"scene_id":self.scene_id,
          "source_subgraph_id":self.source_subgraph_id,"source_component":self.source_component,
          "source_endpoint_id":self.source_node,"source_node":self.source_node,"target_type":self.target_type,
          "target_component_id":self.target_component,"target_component":self.target_component,
          "target_edge_id":self.target_edge_id,"target_node_id":self.target_node_id,
          "target_x":self.target_x,"target_y":self.target_y,"gap_px":self.gap_px,"rank":self.rank,
          "source_category":self.source_category,"features":self.features}

def project_point_to_segment(px,py,ax,ay,bx,by):
    dx,dy=bx-ax,by-ay; den=dx*dx+dy*dy
    t=0.0 if den == 0 else max(0.0,min(1.0,((px-ax)*dx+(py-ay)*dy)/den))
    return ax+t*dx, ay+t*dy

def project_point_to_edge(px,py,edge: Edge):
    best=None
    for (ax,ay),(bx,by) in zip(edge.coordinates,edge.coordinates[1:]):
        qx,qy=project_point_to_segment(px,py,ax,ay,bx,by); d=math.hypot(px-qx,py-qy)
        if best is None or d < best[0]: best=(d,qx,qy)
    return best

def generate_candidates(graph: RoadGraph, scene_id="scene", radius=128.0, top_k=3, main_component=None,
                        support=None, heat_support=None, sources=None, mining_config=None):
    if main_component is None:
        main_component=max(graph.components,key=lambda cid: len(graph.components[cid]),default=None)
    sources = sources or mine_sources(graph, main_component=main_component, **(mining_config or {}))
    rows=[]
    for source in sources:
        for endpoint in source.source_endpoint_ids:
            p=graph.nodes[endpoint]; proposals=[]
            for edge in graph.edges:
                if edge.edge_id in source.source_edge_ids: continue
                # A branch's own attachment neighbourhood is a trivial self-connection.
                if source.attachment_node_id is not None and endpoint == source.attachment_node_id: continue
                d,x,y=project_point_to_edge(p.x,p.y,edge)
                if d <= radius and not (source.attachment_node_id is not None and d <= 8.0 and source.attachment_node_id in {edge.source, edge.target}):
                    proposals.append((d,"edge",edge.edge_id,None,graph.component_id(edge.source),x,y))
            for target_component, target_nodes in graph.components.items():
                for node_id in sorted(graph.endpoints(target_nodes)):
                    if node_id in source.source_nodes: continue
                    q=graph.nodes[node_id]; d=math.hypot(p.x-q.x,p.y-q.y)
                    if d <= radius: proposals.append((d,"endpoint",None,node_id,target_component,q.x,q.y))
            # deterministic dedup: target kind + target identity + rounded projection.
            unique={ (x[1],x[2] or x[3],round(x[5],3),round(x[6],3)):x for x in sorted(proposals) }
            for rank,item in enumerate(sorted(unique.values())[:top_k],1):
                d,kind,eid,nid,tid,x,y=item
                fs={"gap_distance":d,"normalized_gap":d/max(radius,1),"component_nodes":len(source.source_nodes),
                    "component_length_px":source.length_px,"source_degree":graph.graph.degree[endpoint],
                    "target_component_nodes":len(graph.components[tid]),"target_is_main":float(tid==main_component),
                    "gap_graph_support":float(support(p.x,p.y,x,y)) if callable(support) else 0.0,
                    "gap_heat_support":float(heat_support(p.x,p.y,x,y)) if callable(heat_support) else 0.0}
                rows.append(Candidate(f"{scene_id}_{source.source_subgraph_id}_ep_{endpoint}_candidate_{rank:03d}",scene_id,
                    source.source_component_id,endpoint,tid,x,y,d,source.source_type,fs,source.source_subgraph_id,kind,eid,nid,rank))
    return rows
