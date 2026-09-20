"""Generate labeled edge/gap feature datasets from reviewed geographic GT."""
from __future__ import annotations
import argparse, hashlib, json, sys
from functools import lru_cache
from pathlib import Path
import numpy as np
from shapely.geometry import LineString
from graph_utils import load_graph, write_json, xy
from learned_refiner_data import read_gt,validate_splits,edge_label,gap_label,gap_candidates,points_on_line

def extract_features(row: dict,g: object,cache: Path,model_root: Path) -> dict:
    """Use native point sampler and native path extractor with explicit tile scope."""
    import torch
    sys.path.insert(0,str(model_root))
    from model import BilinearSampler,GeodesicPathExtractor
    manifest=json.loads((cache/'manifest.json').read_text())
    if hashlib.sha256((model_root/'model.py').read_bytes()).hexdigest()!=manifest['model_source_sha256']:raise ValueError('Native sampling source differs from cache provenance')
    size=manifest['image_size'];q=points_on_line(row['geometry'],33)
    sampler=BilinearSampler(image_size=512)
    @lru_cache(maxsize=8)
    def tile(name: str) -> dict:
        return torch.load(cache/name,map_location='cpu',weights_only=True)
    values=[];selected=[]
    for point in q:
        choices=[]
        for t in manifest['tiles']:
            local=point-np.asarray(t['origin_xy'])
            if (local>=0).all() and (local<=511).all():choices.append((float(np.min(np.r_[local,511-local])),t,local))
        if not choices:raise ValueError('Sample not covered by cache')
        _,t,local=max(choices,key=lambda c:c[0]);selected.append(t)
        values.append(sampler(tile(t['file'])['encoder_feature'].float()[None],torch.tensor(local,dtype=torch.float32)[None,None])[0,0].numpy())
    f=np.asarray(values);enc=np.r_[f[0],f[-1],f[16],f.mean(0),f.std(0),f.min(0)]
    if row['kind']=='gap':enc=np.r_[enc,abs(f[0]-f[-1]),f[0]*f[-1]]
    road=np.load(cache/'road_probability.npy',mmap_mode='r');indices=np.rint(q).astype(int)
    if (indices<0).any() or (indices[:,0]>=size[0]).any() or (indices[:,1]>=size[1]).any():raise ValueError('Probability index out of range')
    p=road[indices[:,1],indices[:,0]];probs=[p.mean(),np.median(p),np.percentile(p,10),p.std(),p.min(),p.max()]
    # Native path feature only when the entire straight line is inside one tile.
    native=np.zeros(9,dtype=np.float32);applicable=False
    if len(row['geometry'])==2:
        for t in manifest['tiles']:
            local=np.asarray(row['geometry'])-np.asarray(t['origin_xy'])
            if (local>=0).all() and (local<=511).all():
                raw=tile(t['file'])['mask_logits'].float()[None];pts=torch.tensor(local,dtype=torch.float32)
                native=GeodesicPathExtractor(512,32,[1,3,5],5.0)(raw,pts[:1][None],pts[1:][None])[0,0].numpy();applicable=True;break
    u,v=row['u'],row['v'];d=xy(g,v)-xy(g,u);length=LineString(row['geometry']).length;direction=d/max(np.linalg.norm(d),1e-9)
    geo=[length,*direction,g.degree(u),g.degree(v),np.linalg.norm(d)/length]
    if row['kind']=='gap':geo += [row['angle_a'],row['angle_b'],max(row['angle_a'],row['angle_b']),row['component_size_a'],row['component_size_b'],1]
    return {'encoder':enc.astype(float).tolist(),'probability':np.r_[probs,native,float(applicable)].astype(float).tolist(),
            'geometry':np.asarray(geo,dtype=float).tolist(),'native_path_applicable':applicable}

def main() -> None:
    """Fail on coordinate/split/GT ambiguity; do not train or generate pseudo-GT."""
    p=argparse.ArgumentParser();p.add_argument('--splits',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--magroad-root',type=Path);p.add_argument('--labels-only',action='store_true')
    p.add_argument('--positive-distance',type=float,default=8);p.add_argument('--positive-overlap',type=float,default=.9);p.add_argument('--negative-distance',type=float,default=20)
    p.add_argument('--max-candidate-distance',type=float,default=120);p.add_argument('--max-angle',type=float,default=45);a=p.parse_args()
    if not 0<a.positive_overlap<=1 or not 0<a.positive_distance<a.negative_distance or not 20<=a.max_candidate_distance<=120:raise ValueError('Invalid label/candidate thresholds')
    split=json.loads(a.splits.read_text());validate_splits(split)
    if not split['regions']:raise ValueError('No reviewed regions; annotation required')
    if not a.labels_only and a.magroad_root is None:raise ValueError('Native model source required for feature extraction')
    a.output.mkdir(parents=True,exist_ok=True);counts={};rows=[]
    for region in split['regions']:
        graph=load_graph(Path(region['prediction_graph']));size=list(reversed(graph.graph['inference_shape']))
        gt=read_gt(Path(region['gt']),size);x,y,xx,yy=region['bbox_xyxy']
        if gt.graph['region_id']!=region['id']:raise ValueError('GT region mismatch')
        cache=Path(region['cache']) if not a.labels_only else None
        if cache:
            m=json.loads((cache/'manifest.json').read_text())
            if m['units']!='inference_pixel' or m['coordinate_order']!='x,y' or m['image_size']!=size:raise ValueError('Cache mismatch')
            expected=region.get('image_sha256')
            if not expected or m['image_sha256']!=expected:raise ValueError('Region must declare matching source image SHA256')
        existing=[{'kind':'edge','u':u,'v':v,'geometry':e['geometry']} for u,v,e in graph.edges(data=True)]
        gaps=[{'kind':'gap',**r} for r in gap_candidates(graph,a.max_candidate_distance,max_angle=a.max_angle)]
        for row in existing+gaps:
            q=np.asarray(row['geometry'])
            if not ((q[:,0]>=x)&(q[:,0]<=xx)&(q[:,1]>=y)&(q[:,1]<=yy)).all():continue
            label=edge_label(row['geometry'],gt,a.positive_distance,a.positive_overlap,a.negative_distance) if row['kind']=='edge' else gap_label(row,gt,a.positive_distance)
            row.update(label);row.update(region_id=region['id'],split=region['split'],coordinate_order='x,y',units='inference_pixel')
            if cache and row['label']!=-1:row['features']=extract_features(row,graph,cache,a.magroad_root)
            key=f"{region['split']}/{row['kind']}/{row['label']}";counts[key]=counts.get(key,0)+1;rows.append(row)
    write_json(a.output/'samples.json',rows);write_json(a.output/'splits.json',split)
    write_json(a.output/'dataset_statistics.json',{'counts':counts,'rules':vars(a)|{'splits':str(a.splits),'output':str(a.output),'magroad_root':str(a.magroad_root)},'ignored_label':-1,'weighted_BCE':'pos_weight = train negatives / train positives; require both classes before training'})
if __name__=='__main__':main()
