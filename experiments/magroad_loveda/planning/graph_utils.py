"""Future manually reviewed GT interface; no graph extraction here."""
import json
from pathlib import Path


def load_gt_pairs(path):
    data=json.loads(Path(path).read_text())
    if data.get('provenance')!='human_reviewed_gt': raise ValueError('Requires manually reviewed GT pair provenance')
    if data.get('coordinate_order')!='xy': raise ValueError('Requires explicit xy coordinates')
    for pair in data['pairs']:
        if not all(k in pair for k in ('id','start_xy','goal_xy','expected_reachable')): raise ValueError('Incomplete GT pair')
    return data
