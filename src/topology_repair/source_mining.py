"""Deterministic suspicious source-subgraph mining."""
from dataclasses import dataclass
from .graph_io import RoadGraph

@dataclass
class SourceSubgraph:
    source_subgraph_id: str
    source_type: str
    source_component_id: int
    source_nodes: set[int]
    source_edge_ids: set[str]
    source_endpoint_ids: list[int]
    attachment_node_id: int | None
    length_px: float

    def to_dict(self):
        return {"source_subgraph_id":self.source_subgraph_id,"source_type":self.source_type,
                "source_component_id":self.source_component_id,"source_nodes":sorted(self.source_nodes),
                "source_edges":sorted(self.source_edge_ids),"source_endpoint_ids":self.source_endpoint_ids,
                "attachment_node_id":self.attachment_node_id,"length_px":self.length_px,
                "node_count":len(self.source_nodes)}

def _main_component(graph):
    return max(graph.components, key=lambda cid: len(graph.components[cid]), default=None)

def mine_sources(graph: RoadGraph, max_disconnected_component_nodes=64, max_disconnected_component_length_px=2000.0,
                 max_leaf_branch_nodes=64, max_leaf_branch_length_px=1000.0, min_branch_length_px=4.0,
                 main_component=None):
    """Return disconnected components and leaf branches without mutating ``graph``."""
    main = _main_component(graph) if main_component is None else main_component
    out=[]; seen_branches=set()
    for cid, comp in graph.components.items():
        if cid == main or len(comp) > max_disconnected_component_nodes: continue
        length=graph.component_length(comp)
        if length <= max_disconnected_component_length_px:
            out.append(SourceSubgraph(f"source_{len(out):05d}","disconnected_component",cid,set(comp),
                {e.edge_id for e in graph.component_edges(comp)},graph.endpoints(comp),None,length))
    # Each leaf path is endpoint + degree-2 nodes; junction is attachment but not source-owned.
    if main is not None:
        comp=graph.components[main]
        for leaf in sorted(n for n in comp if graph.graph.degree[n] == 1):
            path_nodes=[leaf]; path_edges=[]; previous=None; current=leaf
            while True:
                choices=[v for v in graph.graph.neighbors(current) if v != previous]
                if not choices: break
                nxt=sorted(choices)[0]; edge=next(e for e in graph.edges if {e.source,e.target}=={current,nxt})
                path_edges.append(edge); previous,current=current,nxt; path_nodes.append(current)
                if graph.graph.degree[current] != 2: break
            attachment=path_nodes[-1] if graph.graph.degree[path_nodes[-1]] >= 3 else None
            if attachment is None or len(path_nodes)-1 > max_leaf_branch_nodes: continue
            branch_nodes=set(path_nodes[:-1]); length=sum(e.length() for e in path_edges)
            key=frozenset(e.edge_id for e in path_edges)
            if length < min_branch_length_px or length > max_leaf_branch_length_px or key in seen_branches: continue
            seen_branches.add(key)
            out.append(SourceSubgraph(f"source_{len(out):05d}","leaf_branch",main,branch_nodes,{e.edge_id for e in path_edges},[leaf],attachment,length))
    return out

def inventory(sources): return [s.to_dict() for s in sources]
