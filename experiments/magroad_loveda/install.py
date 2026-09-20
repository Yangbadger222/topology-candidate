"""One guarded additive initialization hook; original configs keep old behavior."""
from pathlib import Path
import difflib
root=Path(__file__).resolve().parent.parent
path=root/'model.py'
old=path.read_text()
needle='        with open(config.SAM_CKPT_PATH, "rb") as f:'
replacement='        # Road-only adaptation loads the complete WildRoad state immediately.\n        if self.config.get("MAGROAD_INIT_ONLY", False):\n            self.matched_param_names = set()\n            return\n'+needle
if 'MAGROAD_INIT_ONLY' not in old:
    assert old.count(needle)==1
    new=old.replace(needle,replacement)
    (Path(__file__).parent/'model_init.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='model.py',tofile='model.py')))
    path.write_text(new)
print('Installed guarded WildRoad initialization hook')
