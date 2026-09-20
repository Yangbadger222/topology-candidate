from .graph_io import RoadGraph

def analyze_components(graph: RoadGraph, main_component: int | None = None) -> list[dict]:
    if main_component is None: main_component = max(graph.components, key=lambda cid: len(graph.components[cid]), default=None)
    out=[]
    for cid, nodes in graph.components.items():
        out.append({"component_id": cid, "nodes": nodes, "is_main": cid == main_component,
                    "node_count": len(nodes), "length_px": graph.component_length(nodes),
                    "endpoints": graph.endpoints(nodes)})
    return out
