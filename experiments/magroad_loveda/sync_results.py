"""Copy experiment reports/artifacts locally; optional follow ends on completion/failure.
Never trains, schedules jobs, changes thresholds, or sends notifications.
"""
from pathlib import Path
import subprocess,sys,time,json,argparse
p=argparse.ArgumentParser(); p.add_argument('--follow',action='store_true'); a=p.parse_args()
repo=Path(__file__).resolve().parents[2]; dest=repo/'outputs/loveda_lora'; dest.mkdir(parents=True,exist_ok=True)
host='badger@100.88.131.52'; remote='/home/badger/sam-inference/MaGRoad/magroad_loveda'
def execute(args): return subprocess.run(args,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True).stdout
while True:
    try:
        execute(['ssh',host,f'/home/badger/overhead-ssl/.venv/bin/python {remote}/tools/create_report.py'])
        # Keep relative report image links valid. Raw full-res probabilities remain remote.
        execute(['rsync','-az','--include=/*.json','--include=/*.md','--include=/review.html','--include=/comparison/','--include=/comparison/***','--exclude=*',host+':'+remote+'/outputs/',str(dest)+'/'])
        for folder in ('baseline','encoder','encoder_decoder','smoke_10','smoke_B'):
            execute(['rsync','-az','--include=/*.json','--exclude=*',host+':'+remote+'/outputs/'+folder+'/',str(dest/folder)+'/']) if folder!='smoke_B' or (dest/'experiment_status.json').exists() and 'experiment_B' in (dest/'experiment_status.json').read_text() else None
        stage=json.loads((dest/'experiment_status.json').read_text())['stage']
        print(time.strftime('%H:%M:%S'),stage,flush=True)
        if not a.follow or stage in ('complete','failed'): break
    except Exception as exc:
        print('Sync error:',repr(exc),flush=True)
        if not a.follow: raise
    time.sleep(60)
