"""Raw and overlay local/context crops for endpoint annotation."""
from pathlib import Path
from PIL import Image, ImageDraw

def _crop(im, cx, cy, size):
    left=int(round(cx-size/2)); top=int(round(cy-size/2)); out=Image.new("RGB",(size,size),(0,0,0))
    box=(max(0,left),max(0,top),min(im.width,left+size),min(im.height,top+size)); out.paste(im.crop(box),(max(0,-left),max(0,-top)))
    return out,left,top

def _draw_polyline(draw, coords, left, top, color, width=3):
    draw.line([(int(x-left),int(y-top)) for x,y in coords],fill=color,width=width,joint="curve")

def render_source_views(image_path, graph, source, endpoint_id, candidates, output_dir, local_size=512, context_size=1024):
    im=Image.open(image_path).convert("RGB"); p=graph.nodes[endpoint_id]
    # center on source endpoint and candidate neighborhood, while preserving fixed dimensions at image borders
    pts=[(p.x,p.y)]+[(c.target_x,c.target_y) for c in candidates]
    cx=sum(x for x,_ in pts)/len(pts); cy=sum(y for _,y in pts)/len(pts); output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    result={}
    for kind,size in (("local",local_size),("context",context_size)):
        raw,left,top=_crop(im,cx,cy,size); overlay=raw.copy(); d=ImageDraw.Draw(overlay)
        src_edges=[e for e in graph.edges if e.edge_id in source.source_edge_ids]
        for e in src_edges: _draw_polyline(d,e.coordinates,left,top,"yellow",4)
        for e in graph.edges:
            if e.edge_id not in source.source_edge_ids: _draw_polyline(d,e.coordinates,left,top,"cyan",2)
        s=(int(p.x-left),int(p.y-top)); d.ellipse((s[0]-6,s[1]-6,s[0]+6,s[1]+6),fill="white"); d.text((s[0]+8,s[1]-12),"S",fill="white")
        for c in candidates:
            t=(int(c.target_x-left),int(c.target_y-top)); d.line([s,t],fill="lime",width=3); d.ellipse((t[0]-5,t[1]-5,t[0]+5,t[1]+5),fill="lime"); d.text((t[0]+7,t[1]-10),f"T{c.rank}",fill="lime")
        for name,img in ((f"raw_{kind}",raw),(f"overlay_{kind}",overlay)):
            path=output_dir/f"{source.source_subgraph_id}_ep_{endpoint_id}_{name}.png"; img.save(path); result[name]=str(path)
    return result

def render_candidate(image_path, graph, candidate, output_path, crop_size=768):
    """Backward-compatible single overlay renderer."""
    from .source_mining import SourceSubgraph
    src=SourceSubgraph(candidate.source_subgraph_id or f"source_{candidate.source_component}",candidate.source_category,candidate.source_component,{candidate.source_node},{candidate.target_edge_id} if candidate.target_edge_id else set(),[candidate.source_node],None,0)
    return render_source_views(image_path,graph,src,candidate.source_node,[candidate],Path(output_path).parent, crop_size,crop_size)["overlay_local"]
