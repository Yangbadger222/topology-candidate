import argparse, json
from .graph_io import load_graph
from .candidate_generation import generate_candidates

def main():
    p=argparse.ArgumentParser(); p.add_argument("graph"); p.add_argument("--scene-id",default="scene"); p.add_argument("--radius",type=float,default=128); p.add_argument("--top-k",type=int,default=3); p.add_argument("--out",default="data/candidates/candidates.json"); a=p.parse_args()
    rows=generate_candidates(load_graph(a.graph),a.scene_id,a.radius,a.top_k)
    import pathlib; pathlib.Path(a.out).parent.mkdir(parents=True,exist_ok=True); json.dump([x.to_dict() for x in rows],open(a.out,"w",encoding="utf-8"),indent=2)
    print(f"generated {len(rows)} candidates -> {a.out}")
if __name__ == "__main__": main()
