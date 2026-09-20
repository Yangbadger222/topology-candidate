#!/usr/bin/env python
"""Summarize normalized topology annotation labels for QA and recall analysis."""
import argparse, json
import re
from collections import Counter
from pathlib import Path

def summarize(data):
    rows = data.get("annotations", data) if isinstance(data, (dict, list)) else []
    if isinstance(rows, dict): rows = list(rows.values())
    sources = {r.get("source_subgraph_id"): r for r in rows}
    out = {"scene_id": data.get("scene_id") if isinstance(data, dict) else None,
           "source_count": len(sources), "endpoint_count": 0,
           "source_validity": dict(Counter(r.get("component_validity", "UNCERTAIN") for r in sources.values())),
           "source_type": dict(Counter(r.get("source_type", "unknown") for r in sources.values())),
           "source_false_reason": dict(Counter(code for r in sources.values() if r.get("component_validity") == "FALSE" for code in r.get("source_reason_codes", []))),
           "endpoint_selection": {}, "candidate_selected_rank": {}, "derived_actions": {}}
    selections = Counter(); ranks = Counter(); actions = Counter()
    for source in sources.values():
        for label in (source.get("endpoint_labels") or {}).values():
            out["endpoint_count"] += 1
            sel = label.get("selection", "UNCERTAIN"); selections["candidate" if sel.startswith("candidate_") else sel] += 1
            if sel.startswith("candidate_"):
                rank = label.get("candidate_rank")
                if rank is None:
                    m = re.search(r"candidate[_-](\d+)$", sel)
                    rank = int(m.group(1)) if m else None
                if rank is not None: ranks[f"T{rank}"] += 1
            actions[label.get("derived_action", "REVIEW")] += 1
    out["endpoint_selection"] = dict(selections); out["candidate_selected_rank"] = dict(ranks); out["derived_actions"] = dict(actions)
    return out

def main():
    p = argparse.ArgumentParser(); p.add_argument("--labels", required=True); p.add_argument("--output")
    a = p.parse_args(); result = summarize(json.loads(Path(a.labels).read_text(encoding="utf-8")))
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if a.output: Path(a.output).write_text(text + "\n", encoding="utf-8")
    print(text)

if __name__ == "__main__": main()
