"""Tensor cache validation; no token truncation or missing top64 placeholders."""
import hashlib
import json
from pathlib import Path
import torch
SCHEMA_VERSION=2

def validate_record(row):
    if row.get('schema_version')!=SCHEMA_VERSION: raise ValueError('cache schema mismatch')
    for name in ('row_id','prompt','response','teacher_logprobs','top64_ids','top64_logprobs','finish_reason','initial_student'):
        if name not in row: raise ValueError('missing field '+name)
    if not isinstance(row['row_id'],int) or row['row_id']<0: raise ValueError('invalid row id')
    for name in ('prompt','response'):
        if not row[name] or any(not isinstance(x,int) or x<0 for x in row[name]): raise ValueError('invalid token ids')
    n=len(row['response'])
    lp,ids,top=(torch.as_tensor(row[k]) for k in ('teacher_logprobs','top64_ids','top64_logprobs'))
    if lp.shape!=(n,) or ids.shape!=(n,64) or top.shape!=(n,64): raise ValueError('cache shape mismatch')
    if not torch.isfinite(lp).all() or not torch.isfinite(top).all(): raise ValueError('nonfinite logprobs')
    if lp.max()>1e-4 or top.max()>1e-4: raise ValueError('positive log probability')
    if ids.dtype not in (torch.int32,torch.int64) or ids.min()<0: raise ValueError('invalid top64 ids')
    if (ids.sort(-1).values.diff(dim=-1)==0).any(): raise ValueError('duplicate candidates')
    if row['finish_reason'] not in ('stop','length'): raise ValueError('invalid finish reason')
    # EOS remains in response when returned by vLLM; all cached positions are active.
    return n

def validate_cache(rows,expected_rows=None):
    ids=[]
    for row in rows: validate_record(row); ids.append(row['row_id'])
    if expected_rows is not None and len(ids)!=expected_rows: raise ValueError('row count mismatch')
    if len(set(ids))!=len(ids): raise ValueError('duplicate row_id')
    return {'schema_version':SCHEMA_VERSION,'rows':len(ids),'row_ids_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest()}

def prompt_hash(prompt): return hashlib.sha256(prompt.encode()).hexdigest()

def read_jsonl(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip(): yield json.loads(line)


def validate_tensor_file(path, expected_rows=None):
    """Exercise weights-only deserialization and probability/ID alignment on disk."""
    path=Path(path)
    records=torch.load(path,map_location='cpu',weights_only=True)
    result=validate_cache(records,expected_rows)
    tokens=longest=matched=0
    for record in records:
        top=record['top64_logprobs']; ids=record['top64_ids'].long()
        response=torch.tensor(record['response'],dtype=torch.long)
        locations=ids==response[:,None]; active=locations.any(-1)
        if (top.exp().sum(-1)>1.0001).any(): raise ValueError('teacher candidate mass exceeds one')
        if active.any():
            selected=top[active].gather(-1,locations[active].long().argmax(-1)[:,None]).flatten()
            if not torch.allclose(selected,record['teacher_logprobs'][active],atol=1e-5,rtol=0):
                raise ValueError('sampled logp disagrees with matching top64 token')
        matched+=int(active.sum()); n=len(response); tokens+=n; longest=max(longest,n)
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''): digest.update(block)
    result.update(tokens=tokens,longest_response=longest,sampled_tokens_in_teacher_top64=matched,file_sha256=digest.hexdigest())
    return result


def main():
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--file',type=Path,required=True)
    parser.add_argument('--expected-rows',type=int); args=parser.parse_args()
    print(json.dumps(validate_tensor_file(args.file,args.expected_rows),allow_nan=False))

if __name__=='__main__': main()
