"""Two-stage, scene-safe baseline training with sklearn when installed."""
from .feature_extraction import feature_matrix

def scene_split(candidates, seed=2026, ratios=(.7,.15,.15)):
    import random
    scenes=sorted({c.scene_id for c in candidates}); random.Random(seed).shuffle(scenes); n=len(scenes); a=int(n*ratios[0]); b=a+int(n*ratios[1]);
    return ([c for c in candidates if c.scene_id in scenes[:a]],[c for c in candidates if c.scene_id in scenes[a:b]],[c for c in candidates if c.scene_id in scenes[b:]])

def train_two_stage(candidates, records, model="random_forest"):
    from sklearn.ensemble import RandomForestClassifier
    by={r["sample_id"]:r for r in records}; labeled=[c for c in candidates if c.sample_id in by]
    real=[c for c in labeled if by[c.sample_id].get("component_validity") in ("REAL","FALSE")]
    conn=[c for c in labeled if by[c.sample_id].get("component_validity")=="REAL" and by[c.sample_id].get("connection_label") in ("CONNECT","NO_CONNECT")]
    def clf(xs, ys):
        m=RandomForestClassifier(n_estimators=100,random_state=2026,class_weight="balanced"); m.fit(feature_matrix(xs),ys); return m
    return {"component":clf(real,[by[c.sample_id]["component_validity"] for c in real]),"connection":clf(conn,[by[c.sample_id]["connection_label"] for c in conn])}
