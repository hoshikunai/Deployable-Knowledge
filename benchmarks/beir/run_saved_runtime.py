"""Run a bounded application-only sample against an existing BEIR runtime."""
from __future__ import annotations
import argparse, json, os, subprocess, time
from pathlib import Path
from urllib import error, request
from app_client import DeployableKnowledgeClient
from run import collapse_chunks_to_documents, select_queries
from beir.datasets.data_loader import GenericDataLoader

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True); p.add_argument('--data-root',type=Path,required=True); p.add_argument('--runtime',type=Path,required=True); p.add_argument('--mapping',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--port',type=int,required=True); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--timeout',type=int,default=180); a=p.parse_args()
    _,queries,_=GenericDataLoader(data_folder=str(a.data_root/a.dataset)).load(split='test'); manifest=json.loads(a.manifest.read_text()); selected=[(q,queries[q]) for q in manifest['queryIds']]
    repository=Path(__file__).resolve().parents[2]
    env={**os.environ,'HOST':'127.0.0.1','PORT':str(a.port),'ORIGIN':f'http://127.0.0.1:{a.port}','BODY_SIZE_LIMIT':'Infinity','DK_MIGRATIONS_DIR':str(repository/'drizzle')}
    log=a.output.with_suffix('.server.log').open('w'); process=subprocess.Popen(['node',str(repository/'build/index.js')],cwd=a.runtime,env=env,stdout=log,stderr=subprocess.STDOUT)
    try:
        deadline=time.monotonic()+120
        while time.monotonic()<deadline:
            try:
                with request.urlopen(f'http://127.0.0.1:{a.port}/heartbeat',timeout=2): break
            except Exception: time.sleep(.5)
        mapping=json.loads(a.mapping.read_text())
        app_to_beir=mapping['applicationToBeir']; client=DeployableKnowledgeClient(f'http://127.0.0.1:{a.port}',search_timeout=a.timeout); rankings={'semantic':{},'hybrid':{}}; failures=[]; timings=[]
        for qid,query in selected:
            started=time.monotonic(); response=None; error_text=None
            for attempt in range(2):
                try: response=client.search(query,10); break
                except Exception as exc: error_text=str(exc)
            timings.append(time.monotonic()-started)
            if response is None:
                failures.append({'queryId':qid,'error':error_text}); response={'semantic':[],'hybrid':[]}
            for method in rankings: rankings[method][qid]=collapse_chunks_to_documents(response.get(method,[]),app_to_beir,10)
        a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps({'queryIds':[q for q,_ in selected],'rankings':rankings},indent=2)+'\n'); (a.output.parent/'failures.json').write_text(json.dumps(failures,indent=2)+'\n'); (a.output.parent/'application-timing.json').write_text(json.dumps({'seconds':sum(timings),'failures':len(failures)},indent=2)+'\n')
    finally:
        if process.poll() is None: process.terminate(); process.wait(timeout=20)
        log.close()
if __name__=='__main__': main()
