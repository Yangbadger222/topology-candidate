"""Sequential full baseline -> Experiment A -> Experiment B; stops on failure.
This is a persistent job, not a scheduler. Never starts topology or A* training.
"""
import subprocess,sys,json,time,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parent
lock=open(ROOT/'experiment.lock','w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
PYTHON=sys.executable
out=ROOT/'outputs'; out.mkdir(exist_ok=True)
state={'started_unix':time.time(),'stage':'initializing','completed':[]}
def status(stage):
    state['stage']=stage; state['updated_unix']=time.time(); (out/'experiment_status.json').write_text(json.dumps(state,indent=2))
def run(stage,args):
    status(stage)
    with open(out/(stage+'.log'),'w') as log:
        subprocess.run([PYTHON,*map(str,args)],cwd=ROOT.parent,stdout=log,stderr=subprocess.STDOUT,check=True)
    state['completed'].append(stage); status(stage+'_complete')
A=ROOT/'config/loveda/remote_encoder.yaml'; B=ROOT/'config/loveda/remote_encoder_decoder.yaml'
base='/home/badger/datasets/overhead_checkpoints/MaGRoad/wildroad_vitb.ckpt'
try:
    run('baseline_target',[ROOT/'train.py','--config',A,'--eval_target_domain','--output-dir',out/'baseline'])
    run('baseline_val',[ROOT/'train.py','--config',A,'--eval-only','--output-dir',out/'baseline'])
    run('experiment_A',[ROOT/'train.py','--config',A])
    run('audit_A',[ROOT/'tools/audit_checkpoint_update.py','--base',base,'--trained',out/'encoder/checkpoints/last.ckpt','--mode','encoder','--output',out/'encoder/parameter_update_audit.json'])
    run('report_after_A',[ROOT/'tools/create_report.py'])
    run('smoke_B',[ROOT/'train.py','--config',B,'--gradient-audit','--max-steps',10,'--limit-train',40,'--limit-val',4,'--output-dir',out/'smoke_B'])
    run('audit_smoke_B',[ROOT/'tools/audit_checkpoint_update.py','--base',base,'--trained',out/'smoke_B/checkpoints/last.ckpt','--mode','encoder_decoder','--output',out/'smoke_B/parameter_update_audit.json'])
    run('experiment_B',[ROOT/'train.py','--config',B])
    run('audit_B',[ROOT/'tools/audit_checkpoint_update.py','--base',base,'--trained',out/'encoder_decoder/checkpoints/last.ckpt','--mode','encoder_decoder','--output',out/'encoder_decoder/parameter_update_audit.json'])
    run('final_report',[ROOT/'tools/create_report.py'])
    status('complete')
except Exception as exc:
    state['error']=repr(exc); status('failed'); raise
