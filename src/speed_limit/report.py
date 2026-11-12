"""Read-only aggregation of persisted receipts, with explicit incompleteness."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def rows(path):
    if not path.exists(): return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(root):
    cache=[json.loads(path.read_text()) for path in sorted((root/'teacher-cache').glob('shard-*/complete.json'))]
    result={'cache_shards_complete':len(cache),'cache_rows':sum(x['rows'] for x in cache),
            'cache_bytes':sum(x['bytes'] for x in cache),'cache_generation_seconds':sum(x['generation_seconds_this_attempt'] for x in cache),
            'cache_teacher_seconds':sum(x['teacher_seconds_this_attempt'] for x in cache),
            'timing_scope':'measured stage wall times; initialization/checkpoint/evaluation excluded from these sums',
            'author_equivalence':'unverified','training':{},'checkpoints':{}}
    for arm in ('opd-calibration','semi'):
        values=rows(root/arm/'train.jsonl')
        result['training'][arm]={'steps':len(values),**{key:sum(x.get(key,0) for x in values) for key in ['generation_seconds','teacher_seconds','update_seconds','refit_seconds']}}
    curve=rows(root/'semi/absorption.jsonl')
    result['absorption_global_semi_diagnostic']={'observations':curve,'tau_resolution_steps':10,'scope':'global training, not repeated S'}
    if curve:
        for percent,threshold in ((50,.5),(90,.1)):
            crossings=[x['step'] for x in curve if x['A']<threshold]
            result['absorption_global_semi_diagnostic'][f'tau_{percent}'] = crossings[0] if crossings else None
            result['absorption_global_semi_diagnostic'][f'tau_{percent}_censored']=not bool(crossings)
    for step in (60,80,100):
        path=root/f'final-step-{step}'/f'eval-step-{step:04d}.json'
        if path.exists(): result['checkpoints'][str(step)]=json.loads(path.read_text())
    result['full_final_eval_complete']=len(result['checkpoints'])==3
    if result['full_final_eval_complete']:
        result['mean_macro_accuracy']=sum(x['mean_accuracy'] for x in result['checkpoints'].values())/3
        result['within_3_points']=abs(result['mean_macro_accuracy']-.398)<=.03
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument('run',type=Path)
    p.add_argument('--watch',action='store_true'); p.add_argument('--detach',action='store_true')
    p.add_argument('--output',type=Path); p.add_argument('--source-revision')
    a=p.parse_args()
    if a.detach:
        if not a.watch or not a.output: raise ValueError('detached mode requires --watch --output')
        argv=[sys.executable,str(Path(__file__).resolve())]+sys.argv[1:]; argv.remove('--detach')
        with (a.run/'summary-publisher.log').open('xb') as log:
            child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        receipt={'pid':child.pid,'source_revision':a.source_revision,'output':str(a.output),'started_at':time.time()}
        (a.run/'summary-publisher.json').write_text(json.dumps(receipt,indent=2))
        print(json.dumps(receipt)); return
    while True:
        value=summarize(a.run)
        value.update(report_source_revision=a.source_revision,updated_at=time.time(),
                     queue_complete=(a.run/'phase0-complete.json').exists(),
                     failed_roles=[path.name for path in a.run.glob('failed-role-*.json')])
        if a.output:
            temporary=a.output.with_suffix('.partial')
            temporary.write_text(json.dumps(value,indent=2,allow_nan=False)); temporary.replace(a.output)
        else: print(json.dumps(value,indent=2,allow_nan=False))
        if not a.watch or value['queue_complete'] or value['failed_roles']: return
        time.sleep(30)

if __name__=='__main__': main()
