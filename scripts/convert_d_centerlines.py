"""Convert human D segments to graph drafts; never auto-approve topology/coverage."""
from __future__ import annotations
import argparse,json
from pathlib import Path

def convert(source: dict) -> dict:
    """Merge exact endpoints only; crossings need explicit human junction review."""
    d={'coordinate_order':'x,y','units':'inference_pixel','image_size':[source['image_size']['width'],source['image_size']['height']],
       'region_id':source['region_id'],'annotation_status':'draft','coverage_complete':False,'nodes':[],'edges':[],'source_image_id':source['image_id']}
    lookup={};seen=set()
    for segment in source['segments']:
        points=segment['points'];ids=[]
        for q in (points[0],points[-1]):
            key=tuple(q)
            if key not in lookup:
                lookup[key]=len(d['nodes']);d['nodes'].append({'id':lookup[key],'x':q[0],'y':q[1]})
            ids.append(lookup[key])
        key=tuple(sorted(ids))
        if ids[0]==ids[1] or key in seen:continue
        seen.add(key);d['edges'].append({'u':ids[0],'v':ids[1],'polyline':points,'path_type':segment['path_type'],'source_edge_id':segment['edge_id'],'visibility':segment.get('visibility'),'confidence':segment.get('confidence')})
    return d

def main() -> None:
    """Convert a folder of existing human annotations without inference labels."""
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    for path in sorted(a.input.glob('*.json')):(a.output/(path.stem+'_graph_gt_draft.json')).write_text(json.dumps(convert(json.loads(path.read_text())),indent=2))
if __name__=='__main__':main()
