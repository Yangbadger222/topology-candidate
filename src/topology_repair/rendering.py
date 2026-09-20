"""Draw all annotation overlays directly on a crop; labels are intentionally absent."""
from pathlib import Path
from PIL import Image, ImageDraw

def render_candidate(image_path, graph, candidate, output_path, crop_size=768):
    im=Image.open(image_path).convert("RGB"); draw=ImageDraw.Draw(im)
    p=graph.nodes[candidate.source_node]; cx,cy=int((p.x+candidate.target_x)/2),int((p.y+candidate.target_y)/2)
    left=max(0,cx-crop_size//2); top=max(0,cy-crop_size//2)
    crop=im.crop((left,top,min(im.width,left+crop_size),min(im.height,top+crop_size)))
    d=ImageDraw.Draw(crop)
    def xy(x,y): return (int(x-left),int(y-top))
    src=graph.components[candidate.source_component]; tgt=graph.components[candidate.target_component]
    for a,b in graph.edges:
        if a in src and b in src: d.line([xy(graph.nodes[a].x,graph.nodes[a].y),xy(graph.nodes[b].x,graph.nodes[b].y)], fill="yellow", width=4)
        if a in tgt and b in tgt: d.line([xy(graph.nodes[a].x,graph.nodes[a].y),xy(graph.nodes[b].x,graph.nodes[b].y)], fill="cyan", width=4)
    s=xy(p.x,p.y); t=xy(candidate.target_x,candidate.target_y)
    d.line([s,t],fill="lime",width=3); d.ellipse((s[0]-6,s[1]-6,s[0]+6,s[1]+6),fill="white"); d.text((s[0]+8,s[1]-12),"S",fill="white")
    d.ellipse((t[0]-6,t[1]-6,t[0]+6,t[1]+6),fill="lime"); d.text((t[0]+8,t[1]-12),"T",fill="lime")
    Path(output_path).parent.mkdir(parents=True,exist_ok=True); crop.save(output_path); return output_path
