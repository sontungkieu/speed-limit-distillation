"""Detached, fail-closed Phase 0 role queues; each role owns one selected GPU."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def save(path,value):
    tmp=path.with_suffix('.partial'); tmp.write_text(json.dumps(value,indent=2)); tmp.replace(path)


def resolve_gpu_index(uuid, inventory):
    matches=[line.split(',')[0].strip() for line in inventory.splitlines()
             if len(line.split(','))==2 and line.split(',')[1].strip()==uuid]
    if len(matches)!=1 or not matches[0].isdigit():
        raise ValueError('GPU UUID must map to exactly one numeric NVML index')
    return matches[0]


def environment(args):
    runtime=args.inputs/'runtime'; cache=args.run/'compile-cache'/f'role-{args.role}'
    short=args.inputs.parent.parent/'tmp'/('sl'+hashlib.sha256(str(args.run).encode()).hexdigest()[:8])/str(args.role)
    short.mkdir(parents=True,exist_ok=True); cache.mkdir(parents=True,exist_ok=True)
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    index=resolve_gpu_index(args.gpu,inventory)
    env=os.environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES':index,'CUDA_DEVICE_ORDER':'PCI_BUS_ID','SPEED_LIMIT_GPU_UUID':args.gpu,'CUDA_HOME':str(runtime/'cuda'),
        'PATH':str(runtime/'cuda/bin')+':'+env.get('PATH',''),'CPLUS_INCLUDE_PATH':str(runtime/'cuda/include/cccl'),
        'PYTHONPATH':str(args.source/'src')+':'+str(runtime/'src'),'NEMO_RL_PREBUILT_ENV_ROOT':str(runtime),
        'TMPDIR':str(short),'TMP':str(short),'TEMP':str(short),'VLLM_RPC_BASE_PATH':str(short),
        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','UV_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false',
        'VLLM_WORKER_MULTIPROC_METHOD':'spawn','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
        'PYTHONUNBUFFERED':'1','RAY_USAGE_STATS_ENABLED':'0','SPEED_LIMIT_REVISION':args.revision})
    for key,directory in [('TRITON_CACHE_DIR','triton'),('TORCH_EXTENSIONS_DIR','torch-extensions'),
        ('FLASHINFER_WORKSPACE_BASE','flashinfer'),('VLLM_CACHE_ROOT','vllm'),('XDG_CACHE_HOME','xdg')]: env[key]=str(cache/directory)
    return env


class Queue:
    def __init__(self,args):
        self.a=args; self.env=environment(args)
        args.run.mkdir(parents=True,exist_ok=True)
        self.state_path=args.run/f'role-{args.role}.json'
        if self.state_path.exists(): raise RuntimeError('Role state exists: inspect; do not duplicate queue')
        self.state={'pid':os.getpid(),'hostname':os.uname().nodename,'role':args.role,'gpu':args.gpu,
                    'source_revision':args.revision,'started_at':time.time(),'phase':'preflight'}
        self.save()
        self.python=str(args.inputs/'runtime/envs/vllm/bin/python')
        self.check_gpu()

    def save(self,**kw):
        self.state.update(kw,heartbeat_at=time.time()); save(self.state_path,self.state)

    def check_gpu(self):
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True)
        matches=[line.split(',') for line in raw.splitlines() if line.split(',')[0].strip()==self.a.gpu]
        if len(matches)!=1 or int(matches[0][1])>1024: raise RuntimeError('Selected GPU UUID absent or occupied')
        processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        if any(line.split(',')[0].strip()==self.a.gpu for line in processes.splitlines()): raise RuntimeError('Selected GPU has a compute process')
        self.save(gpu_preflight=raw)

    def invoke(self,name,task=None,extra=None,command=None):
        args=self.a; log=args.run/(name+'.log')
        argv=command or [self.python,'-m','speed_limit.phase0','--task',task,'--inputs',str(args.inputs),
            '--output',str(args.run/name),'--cache',str(args.run/'teacher-cache')]+(extra or [])
        self.save(phase=name,log=str(log),argv=argv)
        with log.open('xb') as stream:
            child=subprocess.Popen(argv,env=self.env,cwd=args.source,stdout=stream,stderr=subprocess.STDOUT)
            self.save(child_pid=child.pid)
            while child.poll() is None:
                self.save(); time.sleep(10)
        self.save(child_pid=None,last_exit_code=child.returncode)
        if child.returncode: raise RuntimeError(f'{name} failed with exit {child.returncode}; see {log}')
        if task and not (args.run/name/('started.json' if task=='cache' else 'complete.json')).exists(): raise RuntimeError('Missing task receipt')

    def wait(self,paths):
        self.save(phase='waiting',waiting_for=[str(p) for p in paths])
        while not all(path.exists() for path in paths):
            failed=list(self.a.run.glob('failed-role-*.json'))
            if failed: raise RuntimeError('A queue role failed; see '+str(failed))
            self.save(); time.sleep(10)

    def evaluation(self,step):
        checkpoint=self.a.run/'semi'/f'checkpoint-{step:04d}'
        self.wait([checkpoint/'ready.json'])
        self.invoke(f'final-step-{step}','evaluate',['--checkpoint',str(checkpoint),'--checkpoint-step',str(step)])

    def run(self):
        a=self.a
        if a.role==0:
            self.invoke('tests',command=[self.python,'-m','unittest','discover','-s','tests','-v'])
            save(a.run/'tests-passed.json',{'time':time.time(),'revision':a.revision})
            self.invoke('canary','canary',['--steps','2','--batch-size','2','--cap','64','--canary'])
            save(a.run/'canary-passed.json',{'time':time.time(),'revision':a.revision})
            self.invoke('opd-calibration','opd',['--steps','10'])
            self.wait([a.run/'teacher-cache'/f'shard-{i}'/'complete.json' for i in (0,1)])
            self.invoke('semi','train')
            # Evaluator roles take 60, 80 and 100. Role zero computes aggregate receipts.
            self.wait([a.run/f'final-step-{i}'/'complete.json' for i in (60,80,100)]+[a.run/'dev-step-100'/'complete.json'])
            values=[json.loads((a.run/f'final-step-{i}'/f'eval-step-{i:04d}.json').read_text())['mean_accuracy'] for i in (60,80,100)]
            save(a.run/'phase0-complete.json',{'time':time.time(),'source_revision':a.revision,
                'checkpoint_macro_accuracies':dict(zip((60,80,100),values)),
                'mean_macro_accuracy':sum(values)/3,'reference':.398,'within_3_points':abs(sum(values)/3-.398)<=.03,
                'author_equivalence':'unverified','no_phase1_automatically_launched':True})
        else:
            self.wait([a.run/'canary-passed.json'])
            shard=a.role-1
            self.invoke(f'cache-canary-{shard}','cache',['--shard',str(shard),'--limit','2','--cache',str(a.run/f'canary-cache-{shard}')])
            self.invoke(f'cache-{shard}','cache',['--shard',str(shard)])
            for step in ((60,100) if a.role==1 else (80,)): self.evaluation(step)
            if a.role==1:
                self.invoke('dev-step-100','evaluate',['--checkpoint',str(a.run/'semi/checkpoint-0100'),'--checkpoint-step','100','--dev'])
        self.save(phase='complete',finished_at=time.time())


def main():
    parser=argparse.ArgumentParser()
    for name in ('source','inputs','run'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--role',type=int,choices=[0,1,2],required=True)
    parser.add_argument('--gpu',required=True); parser.add_argument('--revision',required=True)
    parser.add_argument('--detach',action='store_true'); args=parser.parse_args()
    args.run.mkdir(parents=True,exist_ok=True)
    if args.detach:
        argv=[sys.executable,'-m','speed_limit.queue']+sys.argv[1:]
        argv.remove('--detach')
        with (args.run/f'controller-{args.role}.log').open('xb') as log:
            child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=args.source)
        print(json.dumps({'controller_pid':child.pid,'role':args.role,'run':str(args.run)})); return
    queue=None
    try:
        queue=Queue(args); queue.run()
    except Exception:
        details={'time':time.time(),'role':args.role,'traceback':traceback.format_exc()}
        save(args.run/f'failed-role-{args.role}.json',details)
        if queue: queue.save(phase='failed',error=details['traceback'])
        raise

if __name__=='__main__': main()
