"""Deterministic SID snapshots and paired evaluation; independent of GPU/models."""
import argparse,hashlib,json,math,random
from pathlib import Path

def create_snapshots(sids,seed,ratios=(0.9,0.7,0.5)):
    universe=sorted(set(sids))
    if not universe or any(not isinstance(s,str) for s in universe):raise ValueError('Nonempty string SID list required')
    order=universe.copy();random.Random(seed).shuffle(order)
    result=[]
    for ratio in ratios:
        if not 0<ratio<=1:raise ValueError('Ratio must be in (0,1]')
        allowed=sorted(order[:int(len(order)*ratio)])
        digest=hashlib.sha256(json.dumps(allowed,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
        result.append({'seed':seed,'ratio':ratio,'count':len(allowed),'allowed':allowed,'sha256':digest})
    return result

def metrics(rows):
 if not rows:return {'denominator':0,'metrics':None}
 ranks=[r['predict'].index(r['output'])+1 if r['output'] in r['predict'] else 1000000 for r in rows]
 return {'denominator':len(rows),'metrics':{key:value for k in (1,3,5,10,20,50) for key,value in [(f'HR@{k}',sum(r<=k for r in ranks)/len(rows)),(f'NDCG@{k}',sum(1/math.log2(r+1) for r in ranks if r<=k)/len(rows))]}}

def summarize(rows,base,allowed):
 # Same canonicalization as the existing official full-catalog metric calculation.
 for collection in (rows,base):
  for row in collection:
   row['output']=row['output'].strip(' \n"')
   row['predict']=[s.strip('"\n').strip() for s in row['predict']]
 assert len(rows)==len(base) and [r['request_id'] for r in rows]==[r['request_id'] for r in base]
 assert all(r['output']==b['output'] for r,b in zip(rows,base))
 selected=[i for i,r in enumerate(rows) if r['output'] in allowed]
 invalid=sum(s not in allowed for r in rows for s in r['predict'])
 assert invalid==0,'Unavailable SID returned'
 return {'all_requests':len(rows),'available_targets':len(selected),'unavailable_targets':len(rows)-len(selected),'unavailable_target_fraction':1-len(selected)/len(rows),'paired_dynamic':metrics([rows[i] for i in selected]),'paired_full_catalog':metrics([base[i] for i in selected]),'invalid_predictions':invalid,'total_predictions':sum(len(r['predict']) for r in rows),'topk_shortfall_requests':{str(k):sum(len(set(r['predict']))<k for r in rows) for k in (5,10,50)}}

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('create');c.add_argument('--sids',type=Path,required=True);c.add_argument('--seed',type=int,default=2026);c.add_argument('--output',type=Path,required=True)
    e=sub.add_parser('evaluate');e.add_argument('--full',type=Path,required=True);e.add_argument('--subset',type=Path,required=True);e.add_argument('--allowed',type=Path,required=True);e.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    def load(path):return json.loads(path.read_text(encoding='utf-8'))
    if a.command=='create':result=create_snapshots(load(a.sids),a.seed)
    else:
        full,subset=load(a.full),load(a.subset)
        if not full:raise ValueError('Nonempty request set required')
        for rows in (full,subset):
            if len({r['request_id'] for r in rows})!=len(rows):raise ValueError('Duplicate request IDs')
        result=summarize(subset,full,set(load(a.allowed)))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
