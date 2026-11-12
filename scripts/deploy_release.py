"""Fetch a hash-pinned public source release via proxy, verify, then launch a role."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    for key in ('url','sha256','revision','proxy','gpu'): p.add_argument('--'+key,required=True)
    for key in ('source','run','inputs'): p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--stage-only',action='store_true')
    p.add_argument('--role',type=int,required=True); a=p.parse_args()
    write_root=Path('/workspace/storage-shared/nlp/tungks').resolve()
    for path in (a.source,a.run):
        if not path.resolve().is_relative_to(write_root) or path.resolve()==write_root: raise ValueError('Output escapes assigned project root')
    receipt=a.source/'release-receipt.json'
    if receipt.exists():
        value=json.loads(receipt.read_text())
        if value['sha256']!=a.sha256 or value['revision']!=a.revision: raise ValueError('Existing source identity mismatch')
        for name,expected in value['files'].items():
            if digest(a.source/name)!=expected: raise ValueError('Source was modified: '+name)
    else:
        a.source.mkdir(parents=True,exist_ok=False)
        archive=a.source.parent/(a.source.name+'.tar.gz')
        transport=archive.with_suffix('.download') if '.json' in a.url.split('?')[0] else archive
        subprocess.run(['curl','--fail','--location','--retry','4','--retry-all-errors','--retry-delay','2',
                        '--connect-timeout','20','--max-time','120','--max-filesize','10485760',
                        '--proxy',a.proxy,'--output',str(transport),a.url],check=True)
        if transport!=archive:
            bundle=json.loads(transport.read_text())
            if bundle['source_revision']!=a.revision or bundle['archive_sha256']!=a.sha256:
                raise ValueError('Source bundle identity mismatch')
            data=base64.b64decode(bundle['archive_base64'],validate=True)
            if hashlib.sha256(data).hexdigest()!=a.sha256: raise ValueError('Decoded archive checksum mismatch')
            with archive.open('xb') as stream: stream.write(data)
        if digest(archive)!=a.sha256: raise ValueError('Source archive SHA-256 mismatch')
        with tarfile.open(archive) as tar:
            for item in tar.getmembers():
                if not (a.source/item.name).resolve().is_relative_to(a.source.resolve()) or item.issym() or item.islnk(): raise ValueError('Unsafe archive member')
            tar.extractall(a.source,filter='data')
        files={str(path.relative_to(a.source)):digest(path) for path in a.source.rglob('*') if path.is_file()}
        receipt.write_text(json.dumps({'revision':a.revision,'sha256':a.sha256,'url':a.url,'files':files},indent=2))
    if a.stage_only:
        print(json.dumps({'source':str(a.source),'revision':a.revision,'sha256':a.sha256,'staged':True})); return
    env=os.environ.copy(); env['PYTHONPATH']=str(a.source/'src')+':'+str(a.inputs/'runtime/src')
    subprocess.run([str(a.inputs/'runtime/envs/vllm/bin/python'),'-m','speed_limit.queue','--source',str(a.source),
        '--inputs',str(a.inputs),'--run',str(a.run),'--role',str(a.role),'--gpu',a.gpu,'--revision',a.revision,'--detach'],env=env,check=True)

if __name__=='__main__': main()
