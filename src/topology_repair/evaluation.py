import json
from collections import Counter
from .derive_action import derive_action

def classification_metrics(y_true, y_pred, positive=1):
    tp=sum(a==positive and b==positive for a,b in zip(y_true,y_pred)); fp=sum(a!=positive and b==positive for a,b in zip(y_true,y_pred)); fn=sum(a==positive and b!=positive for a,b in zip(y_true,y_pred))
    p=tp/(tp+fp) if tp+fp else 0.; r=tp/(tp+fn) if tp+fn else 0.
    return {"precision":p,"recall":r,"f1":2*p*r/(p+r) if p+r else 0.}

def graph_stats(graph):
    import networkx as nx
    return {"connected_components":nx.number_connected_components(graph.graph),"dangling_endpoints":sum(graph.graph.degree[n]==1 for n in graph.graph)}

def evaluate_records(records, graph_before=None, graph_after=None):
    valid=[r for r in records if r.get("component_validity") in ("REAL","FALSE")]
    false_true=[r["component_validity"]=="FALSE" for r in valid]; false_pred=[r.get("pred_component_validity")=="FALSE" for r in valid]
    out={"component_false":classification_metrics(false_true,false_pred,True)}
    conn=[r for r in records if r.get("component_validity")=="REAL" and r.get("connection_label") in ("CONNECT","NO_CONNECT")]
    out["connection"] = classification_metrics([r["connection_label"]=="CONNECT" for r in conn],[r.get("pred_connection_label")=="CONNECT" for r in conn],True)
    out["false_connection_rate"] = sum(r.get("pred_connection_label")=="CONNECT" and r["connection_label"]=="NO_CONNECT" for r in conn)/len(conn) if conn else 0.
    if graph_before: out["before"]=graph_stats(graph_before)
    if graph_after: out["after"]=graph_stats(graph_after)
    return out

def save_report(report,path):
    with open(path,"w",encoding="utf-8") as f: json.dump(report,f,indent=2)
