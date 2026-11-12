"""Phase 0 runner. Independent reconstruction; author equivalence unverified."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import random
import time
import torch
from .legacy import Experiment, append_json, write_json, check_objective
from .cache import validate_record


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''): digest.update(block)
    return digest.hexdigest()


def distribution(model, record, teacher=False, gradients=False):
    """Full-vocabulary sampled logp and fixed-support top64 logp, aligned to response."""
    prompt, response = record['prompt'], record['response']
    if not prompt or not response: raise ValueError('empty prompt/response')
    device = next(model.parameters()).device
    ids = torch.tensor([prompt + response], device=device, dtype=torch.long)
    sampled, candidates, indices = [], [], []
    with (nullcontext() if gradients else torch.no_grad()), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type=='cuda'):
        logits = model(input_ids=ids, use_cache=False).logits[0, len(prompt)-1:-1]
        if logits.shape[0] != len(response): raise RuntimeError('response/logit alignment mismatch')
        for start in range(0, len(response), 128):
            chunk = logits[start:start+128].float()
            logz = chunk.logsumexp(-1, keepdim=True)
            labels = ids[0, len(prompt)+start:len(prompt)+start+128]
            sampled.append((chunk.gather(-1, labels[:, None])-logz).squeeze(-1))
            if teacher:
                top, index = chunk.topk(64, dim=-1)
                indices.append(index.to('cpu', dtype=torch.int32)); candidates.append((top-logz).cpu())
            else:
                index = record['top64_ids'][start:start+128].to(device, dtype=torch.long)
                candidates.append(chunk.gather(-1, index)-logz)
        result = torch.cat(sampled)
        if teacher: return result.cpu(), torch.cat(indices), torch.cat(candidates)
        return result, torch.cat(candidates)


def top64_kl(student, teacher):
    student = student.float().log_softmax(-1)
    teacher = teacher.to(student.device).float().log_softmax(-1)
    return (student.exp()*(student-teacher)).sum(-1)


class Phase0(Experiment):
    def __init__(self, args):
        import numpy as np
        import pyarrow.parquet as pq
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from vllm import LLM, SamplingParams
        args.run_id = args.output.name
        self.args, self.torch, self.SamplingParams = args, torch, SamplingParams
        self.root, self.run = args.inputs, args.output
        self.run.mkdir(parents=True, exist_ok=True)
        if (self.run/'started.json').exists(): raise RuntimeError('Attempt exists; inspect before selecting a fresh output')
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.set_num_threads(4)
        self.started = time.time(); self.objective_check = check_objective(torch)
        self.data = pq.read_table(self.root/'data/train.parquet').to_pylist()
        if len(self.data)!=14116 or len({r['prompt'] for r in self.data})!=14116: raise RuntimeError('Expected 14116 unique prompts')
        if sha256(self.root/'data/train.parquet')!='40f672fe8b6dbeee953ae5acc75a38e99144e682450e07c625c87c734ec76285': raise RuntimeError('Pinned parquet mismatch')
        if sha256(self.root/'models/student/tokenizer.json')!=sha256(self.root/'models/teacher/tokenizer.json'): raise RuntimeError('Tokenizer mismatch')
        self.tokenizer = AutoTokenizer.from_pretrained(self.root/'models/student',local_files_only=True)
        self.order = list(range(len(self.data))); random.Random(args.seed).shuffle(self.order)
        write_json(self.run/'prompt_order.json', self.order)
        self.benchmarks = {name:pq.read_table(self.root/'data'/f'{name}.parquet').to_pylist() for name in ['aime24','aime25','aime26','hmmt26']}
        self.batch_size, self.cap, self.steps = args.batch_size,args.cap,args.steps
        self.snapshot, self.last_refit = self.run/'rollout_weights.pt',0
        self.cache_directory = args.cache
        write_json(self.run/'started.json',{**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'pid':os.getpid(),'started_at':self.started,'torch':torch.__version__,'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'objective_check':self.objective_check,'source_revision':os.environ.get('SPEED_LIMIT_REVISION'),'author_implementation_equivalence':'unverified','training_backend':'FP32 master / BF16 compute'})
        print('INITIALIZING_VLLM',args.task,flush=True)
        self.engine = LLM(model=str(args.checkpoint or self.root/'models/student'),tokenizer=str(self.root/'models/student'),dtype='bfloat16',tensor_parallel_size=1,enforce_eager=True,gpu_memory_utilization=0.85 if args.task=='evaluate' else 0.36,max_model_len=34816,max_num_seqs=16,max_num_batched_tokens=8192,enable_prefix_caching=False,disable_log_stats=True,generation_config='vllm',worker_extension_cls='speed_limit.worker_extension.RefitExtension',seed=args.seed)
        if args.task=='evaluate': return
        print('INITIALIZING_TEACHER',flush=True)
        self.teacher = AutoModelForCausalLM.from_pretrained(self.root/'models/teacher',dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
        self.teacher.requires_grad_(False)
        if args.task in ('train','opd','canary'):
            self.student = AutoModelForCausalLM.from_pretrained(self.root/'models/student',dtype=torch.float32,attn_implementation='sdpa',local_files_only=True).to('cuda')
            self.student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False}); self.student.train()
            self.optimizer = torch.optim.AdamW(self.student.parameters(),lr=1e-6,betas=(0.9,0.999),eps=1e-8,weight_decay=0.01,fused=True)

    def score(self, records):
        started=time.monotonic()
        for record in records:
            lp,ids,top=distribution(self.teacher,record,teacher=True)
            record.update(schema_version=2,teacher_logprobs=lp,top64_ids=ids,top64_logprobs=top)
            validate_record(record)
        torch.cuda.synchronize(); return time.monotonic()-started

    def collect(self,row_ids,step,initial=False):
        started=time.monotonic()
        prompts=[self.render(self.data[index]['prompt'],training=True) for index in row_ids]
        outputs=self.generate(prompts,self.cap,self.args.seed*100000+step)
        self.generation_seconds=time.monotonic()-started
        records=[{'row_id':index,'prompt':prompt,'response':list(output.outputs[0].token_ids),'finish_reason':output.outputs[0].finish_reason,'initial_student':initial} for index,prompt,output in zip(row_ids,prompts,outputs)]
        self.scoring_seconds=self.score(records); return records

    def build_cache(self):
        directory=self.cache_directory/f'shard-{self.args.shard}'; directory.mkdir(parents=True,exist_ok=True)
        selected=self.order[self.args.shard::self.args.shards]
        if self.args.limit: selected=selected[:self.args.limit]
        gen=score=count=0; entries=[]
        for offset in range(0,len(selected),32):
            path=directory/f'batch-{offset//32:05d}.pt'; rows=selected[offset:offset+32]
            if path.exists():
                records=torch.load(path,map_location='cpu',weights_only=True)
                if [r['row_id'] for r in records]!=rows: raise RuntimeError('Cache row order mismatch')
                for record in records: validate_record(record)
            else:
                records=self.collect(rows,offset//32,initial=True)
                partial=path.with_suffix('.partial'); torch.save(records,partial); partial.replace(path)
                gen+=self.generation_seconds; score+=self.scoring_seconds
            entries.append({'file':str(path.relative_to(self.cache_directory)),'sha256':sha256(path),'rows':rows}); count+=len(rows)
            append_json(self.run/'cache_progress.jsonl',{'rows':count,'total':len(selected),'generation_seconds':gen,'teacher_seconds':score,'time':time.time()})
            print('CACHE',self.args.shard,count,len(selected),flush=True)
        write_json(directory/'complete.json',{'schema_version':2,'shard':self.args.shard,'shards':self.args.shards,'seed':self.args.seed,'cap':self.cap,'rows':count,'entries':entries,'generation_seconds_this_attempt':gen,'teacher_seconds_this_attempt':score,'bytes':sum((self.cache_directory/e['file']).stat().st_size for e in entries)})

    def measurement(self,records,step,label):
        self.student.eval(); total=tokens=0; per_rollout=[]
        with torch.no_grad():
            for record in records:
                _,current=distribution(self.student,record)
                kl=top64_kl(current,record['top64_logprobs'])
                if not torch.isfinite(kl).all() or kl.min() < -2e-5: raise RuntimeError('Invalid top64 KL')
                total+=float(kl.sum()); tokens+=kl.numel(); per_rollout.append(float(kl.mean()))
        result={'step':step,'set':label,'mean_kl_per_token':total/tokens,'per_rollout_kl':per_rollout,'rollouts':len(records),'tokens':tokens,'time':time.time()}
        append_json(self.run/'kl.jsonl',result); self.student.train(); return result

    def update(self,records):
        started=time.monotonic(); self.student.train(); self.optimizer.zero_grad(set_to_none=True); losses=[]
        for record in records:
            current,_=distribution(self.student,record,gradients=True)
            teacher=record['teacher_logprobs'].to(current.device)
            if current.shape!=teacher.shape: raise RuntimeError('Token alignment mismatch')
            loss=-((teacher-current.detach()).detach()*current).sum()/len(records)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            loss.backward(); losses.append(float(loss.detach()))
        norm=torch.nn.utils.clip_grad_norm_(self.student.parameters(),1.,error_if_nonfinite=True)
        self.optimizer.step(); torch.cuda.synchronize()
        return {'loss':sum(losses),'gradient_norm':float(norm),'update_seconds':time.monotonic()-started,'mean_response_tokens':sum(len(r['response']) for r in records)/len(records),'cap_hit_fraction':sum(r['finish_reason']=='length' for r in records)/len(records)}

    def run_training(self):
        is_opd=self.args.task in ('opd','canary')
        if not is_opd:
            index,splits=load_cache(self.cache_directory,self.args.shards,self.order)
            fixed=load_rows(index,splits['state_sets'][0]); heldout=load_rows(index,splits['heldout'])
            baseline=self.measurement(fixed,0,'S0')['mean_kl_per_token']; self.measurement(heldout,0,'heldout')
            if baseline<=0: raise RuntimeError('Nonpositive baseline KL')
        for step in range(1,self.steps+1):
            rows=self.order[(step-1)*self.batch_size:step*self.batch_size]
            if len(rows)!=self.batch_size: raise RuntimeError('Horizon exceeds first epoch')
            if is_opd:
                refit_started=time.monotonic(); self.refit(step-1)
                refit_seconds=time.monotonic()-refit_started; records=self.collect(rows,step-1)
            else:
                records=load_rows(index,rows); self.generation_seconds=self.scoring_seconds=0; refit_seconds=0
            result=self.update(records); result.update(step=step,refit_seconds=refit_seconds,generation_seconds=self.generation_seconds,teacher_seconds=self.scoring_seconds,time=time.time())
            append_json(self.run/'train.jsonl',result); print('TRAIN_STEP',step,result,flush=True)
            if not is_opd and (step%self.args.measure_every==0 or step==self.steps):
                metric=self.measurement(fixed,step,'S0')
                append_json(self.run/'absorption.jsonl',{'step':step,'A':metric['mean_kl_per_token']/baseline,'scope':'fixed diagnostic S0 under global Semi training; not repeated-S training'})
                self.measurement(heldout,step,'heldout')
            if not is_opd and step%50==0:
                self.refit(step); fresh=self.collect(splits['heldout'][:128],100000+step)
                self.measurement(fresh,step,'fresh')
                append_json(self.run/'fresh_costs.jsonl',{'step':step,'generation_seconds':self.generation_seconds,'teacher_seconds':self.scoring_seconds,'mean_response_tokens':sum(len(r['response']) for r in fresh)/len(fresh)})
            if step in (60,80,100) or step==self.steps:
                self.checkpoint(step)
                write_json(self.run/f'checkpoint-{step:04d}'/'ready.json',{'step':step,'time':time.time()})
        write_json(self.run/'complete.json',{'steps':self.steps,'task':self.args.task,'time':time.time(),'elapsed_seconds':time.time()-self.started,'final_benchmark_evaluation_complete':False})

    def run_evaluation(self):
        step=self.args.checkpoint_step; self.last_refit=step
        if self.args.dev: self.benchmarks={name:rows for name,rows in self.benchmarks.items() if name in ('aime24','aime25')}
        self.evaluate(step)
        write_json(self.run/'complete.json',{'step':step,'time':time.time(),'final_eval':True})


def load_cache(directory,shards,order):
    index={}
    for shard in range(shards):
        manifest=json.loads((directory/f'shard-{shard}'/'complete.json').read_text())
        if manifest['shards']!=shards or manifest['seed']!=0 or manifest['cap']!=4096: raise RuntimeError('Cache contract mismatch')
        for entry in manifest['entries']:
            file=directory/entry['file']
            if sha256(file)!=entry['sha256']: raise RuntimeError('Cache hash mismatch')
            for row in entry['rows']:
                if row in index: raise RuntimeError('Duplicate cache row')
                index[row]=file
    if set(index)!=set(order): raise RuntimeError('Cache incomplete')
    pool,heldout=order[:2048],order[2048:2304]
    splits={'pool':pool,'heldout':heldout,'state_sets':[random.Random(seed).sample(pool,64) for seed in (0,1,2)],'seed':0,'heldout_scope':'disjoint from P; global Phase0 Semi trains all prompts'}
    path=directory/'splits.json'
    if path.exists() and json.loads(path.read_text())!=splits: raise RuntimeError('Split mismatch')
    if not path.exists(): write_json(path,splits)
    return index,splits


def load_rows(index,rows):
    files={index[row] for row in rows}; loaded={}
    for file in files:
        for record in torch.load(file,map_location='cpu',weights_only=True):
            if record['row_id'] in rows: validate_record(record); loaded[record['row_id']]=record
    return [loaded[row] for row in rows]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--task',choices=['cache','train','opd','canary','evaluate'],required=True)
    for name in ('inputs','output','cache'): parser.add_argument('--'+name,type=Path,required=True)
    for name,default in [('shard',0),('shards',2),('seed',0),('cap',4096),('batch-size',128),('steps',100),('limit',0),('measure-every',10),('checkpoint-step',0)]: parser.add_argument('--'+name,type=int,default=default)
    parser.add_argument('--dev',action='store_true')
    parser.add_argument('--checkpoint',type=Path); parser.add_argument('--canary',action='store_true')
    args=parser.parse_args(); runner=Phase0(args)
    if args.task=='cache': runner.build_cache()
    elif args.task=='evaluate': runner.run_evaluation()
    else: runner.run_training()

if __name__=='__main__': main()
