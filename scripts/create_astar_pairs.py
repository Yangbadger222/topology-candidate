"""Generate a local HTML human annotation tool; coordinates remain inference xy."""
from __future__ import annotations
import argparse, base64, io, json
from pathlib import Path
from PIL import Image

def main() -> None:
    """Click centerline nodes/paths or genuine test start/goal pairs and export JSON."""
    p=argparse.ArgumentParser();p.add_argument('--image',type=Path,required=True);p.add_argument('--image-size',type=int,nargs=2,required=True)
    p.add_argument('--region',required=True);p.add_argument('--split',choices=['train','validation','test'],required=True)
    p.add_argument('--mode',choices=['pairs','graph'],default='pairs');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    im=Image.open(a.image).convert('RGB').resize(tuple(a.image_size),Image.Resampling.LANCZOS);b=io.BytesIO();im.save(b,format='JPEG',quality=95)
    info={'coordinate_order':'x,y','units':'inference_pixel','image_size':a.image_size,'region_id':a.region,'split':a.split,
          'annotation_status':'draft','coverage_complete':False,'nodes':[],'edges':[]}
    html='''<!doctype html><meta charset="utf-8"><title>人工道路标注</title>
<style>body{font-family:sans-serif;background:#222;color:white}canvas{cursor:crosshair}button{margin:8px}#wrap{overflow:auto;height:80vh}</style>
<p>坐标：inference_pixel x,y。滚动查看，按显示比例定位；邻近节点6px自动复用，只有点击同一节点才表示路口。请先阅读标注规范。</p>
<button onclick="end()">结束当前道路 / 取消未完成pair</button><button onclick="undo()">撤销</button><button onclick="download()">导出人工草稿</button>
<label>道路类型 <select id="ptype"><option>pedestrian_path</option><option>vehicle_road</option><option>service_path</option><option>narrow_path</option></select></label>
<label>Pair类别 <select id="category"><option>main_road</option><option>cross_block</option><option>tree_occluded</option><option>campus_internal</option><option>long_multi_junction</option></select></label>
<label>人工可达 <select id="reachable"><option value="true">是</option><option value="false">否</option></select></label><label>显示比例 <input type="range" min="20" max="150" value="50" oninput="zoom(this.value)"></label><span id="status"></span>
<div id="wrap"><canvas id="c"></canvas></div><script>
const meta=METADATA, mode=MODE, img=new Image(),c=document.getElementById('c'),ctx=c.getContext('2d');
let current=null,pairs=[],history=[];img.src='data:image/jpeg;base64,IMAGE';img.onload=()=>{c.width=img.width;c.height=img.height;zoom(50);draw()};
function zoom(v){c.style.width=img.width*v/100+'px';c.style.height=img.height*v/100+'px'}
function end(){current=null;draw()}
function undo(){if(history.length){let s=history.pop();meta.nodes=s.nodes;meta.edges=s.edges;pairs=s.pairs;current=s.current;draw()}}
c.onclick=e=>{history.push(JSON.parse(JSON.stringify({nodes:meta.nodes,edges:meta.edges,pairs,current})));let r=c.getBoundingClientRect(),x=(e.clientX-r.left)*c.width/r.width,y=(e.clientY-r.top)*c.height/r.height;
if(mode==='pairs'){if(current===null)current=[x,y];else{pairs.push({id:'pair_'+pairs.length,start:current,goal:[x,y],category:document.getElementById('category').value,expected_reachable:document.getElementById('reachable').value==='true',region_id:meta.region_id,split:meta.split,coordinate_order:'x,y',units:'inference_pixel',image_size:meta.image_size});current=null}}
else{let n=meta.nodes.find(n=>Math.hypot(n.x-x,n.y-y)<=6);if(!n){n={id:meta.nodes.length,x,y};meta.nodes.push(n)}
if(current!==null&&current!==n.id&&!meta.edges.some(e=>(e.u===current&&e.v===n.id)||(e.v===current&&e.u===n.id))){let u=meta.nodes[current];meta.edges.push({u:current,v:n.id,polyline:[[u.x,u.y],[n.x,n.y]],path_type:document.getElementById('ptype').value})}current=n.id}draw()};
function draw(){ctx.drawImage(img,0,0);ctx.lineWidth=3;ctx.strokeStyle='#00ffcc';for(let e of meta.edges){ctx.beginPath();e.polyline.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.stroke()}
for(let n of meta.nodes){ctx.fillStyle='yellow';ctx.beginPath();ctx.arc(n.x,n.y,4,0,7);ctx.fill()}
for(let p of pairs){ctx.strokeStyle='#ff66ff';ctx.beginPath();ctx.moveTo(...p.start);ctx.lineTo(...p.goal);ctx.stroke()}
if(mode==='pairs'&&current){ctx.fillStyle='yellow';ctx.fillRect(current[0]-4,current[1]-4,8,8)}document.getElementById('status').textContent='节点 '+meta.nodes.length+' 边 '+meta.edges.length+' pairs '+pairs.length}
function download(){let data=mode==='pairs'?pairs:meta,url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download=mode==='pairs'?'astar_pairs.json':meta.region_id+'_graph_gt_draft.json';a.click();URL.revokeObjectURL(url)}
</script>'''
    html=html.replace('METADATA',json.dumps(info)).replace('MODE',json.dumps(a.mode)).replace('IMAGE',base64.b64encode(b.getvalue()).decode())
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(html)
if __name__=='__main__':main()
