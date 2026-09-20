"""Audit genuine manual test pairs and compare available XJTLU graph baselines."""
from __future__ import annotations
import argparse, csv, hashlib, json, shutil
from pathlib import Path
import numpy as np
from PIL import ImageDraw
from graph_utils import load_graph, background, overlay, stats, write_json
from evaluate_navigation_graph import route

def main() -> None:
    """Retain invalid records, exclude them explicitly, never silently clamp input."""
    p=argparse.ArgumentParser();p.add_argument('--pairs',type=Path,required=True);p.add_argument('--raw',type=Path,required=True);p.add_argument('--rule',type=Path,required=True);p.add_argument('--image',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);shutil.copyfile(a.pairs,a.output/'astar_pairs_original.json')
    graphs={'raw':load_graph(a.raw),'rule':load_graph(a.rule)};shape=graphs['raw'].graph['inference_shape'];h,w=shape
    if graphs['rule'].graph['inference_shape']!=shape:raise ValueError('Graph frames differ')
    pairs=json.loads(a.pairs.read_text());ids=set();audit=[];valid=[]
    for pair in pairs:
        errors=[];identifier=pair.get('id')
        if not isinstance(identifier,str) or Path(identifier).name!=identifier or identifier in ids:errors.append('invalid/duplicate ID')
        ids.add(identifier)
        if pair.get('coordinate_order')!='x,y' or pair.get('units')!='inference_pixel' or pair.get('image_size')!=[w,h]:errors.append('coordinate frame mismatch')
        if pair.get('split')!='test' or pair.get('region_id')!='xjtlu_1':errors.append('not XJTLU test pair')
        for key in ('start','goal'):
            q=np.asarray(pair.get(key,[]),dtype=float)
            if q.shape!=(2,) or not np.isfinite(q).all() or not (0<=q[0]<w and 0<=q[1]<h):errors.append(key+' out of bounds/invalid')
        if not isinstance(pair.get('expected_reachable'),bool):errors.append('manual expected_reachable required')
        audit.append({'id':identifier,'valid':not errors,'errors':errors})
        if not errors:valid.append(pair)
    write_json(a.output/'pair_audit.json',audit);write_json(a.output/'astar_pairs_valid.json',valid)
    base=background(a.image,shape);results=[];summaries={};examples={};preview=base.copy();draw=ImageDraw.Draw(preview)
    for pair in valid:
        for key,color in [('start','#00ff66'),('goal','#ff66ff')]:
            x,y=pair[key];draw.ellipse((x-5,y-5,x+5,y+5),fill=color);draw.text((x+6,y),pair['id'],fill=color)
        row={'pair':pair}
        for name,g in graphs.items():
            answer,temporary,lines=route(g,pair,40);row[name]=answer
            tag='success' if answer['success'] else ('projection_failure' if 'start_projection' not in answer else 'disconnected')
            if (name,tag) not in examples:
                examples[name,tag]=pair['id'];path=a.output/f'{name}_{tag}_{pair["id"]}.png'
                overlay(temporary,base,path,lines=lines)
                im=__import__('PIL.Image',fromlist=['Image']).open(path);d=ImageDraw.Draw(im)
                for key,color in [('start','#00ff66'),('goal','#ff66ff')]:
                    x,y=pair[key];d.ellipse((x-8,y-8,x+8,y+8),fill=color)
                im.thumbnail((1800,1300));im.save(path)
        results.append(row)
    preview.thumbnail((1800,1300));preview.save(a.output/'manual_pairs_overview.jpg',quality=95)
    positive=sum(p['expected_reachable'] for p in valid)
    for name,g in graphs.items():
        answers=[r[name] for r in results];success=sum(r['success'] for r in answers)
        lengths=[r['path_length'] for r in answers if r['success']]
        projected=[r for r in answers if 'start_projection' in r]
        summaries[name]={'valid_pairs':len(valid),'success':success,'success_rate':success/len(valid) if valid else None,
            'expected_reachable_pairs':positive,'reachable_pair_success_rate':sum(r[name]['success'] and r['pair']['expected_reachable'] for r in results)/positive if positive else None,
            'projection_failures':len(valid)-len(projected),'disconnected_after_projection':sum(not r['success'] for r in projected),
            'connected_pair_ratio':success/len(valid) if valid else None,'mean_successful_path_length':float(np.mean(lengths)) if lengths else None,'path_length_units':'inference_pixel','graph':stats(g)}
    write_json(a.output/'results.json',results)
    config={'max_projection_distance':40,'cost':'pure_distance','uncertainty_evaluation':'unavailable: baseline graph has no calibrated learned confidence','source_pairs_sha256':hashlib.sha256(a.pairs.read_bytes()).hexdigest(),'input_pairs':len(pairs),'invalid_pairs':len(pairs)-len(valid),'summary':summaries}
    write_json(a.output/'summary.json',config)
    with (a.output/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['graph','valid_pairs','success','success_rate','projection_failures','disconnected_after_projection','mean_successful_path_length']);writer.writeheader()
        for name,r in summaries.items():writer.writerow({'graph':name,**{k:r[k] for k in writer.fieldnames if k!='graph'}})
    print(json.dumps({name:{k:r[k] for k in ('valid_pairs','success','success_rate','projection_failures','disconnected_after_projection')} for name,r in summaries.items()},indent=2))
if __name__=='__main__':main()
