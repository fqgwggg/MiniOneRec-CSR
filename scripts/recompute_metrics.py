"""CPU-only results reader; recompute only predictions actually supplied."""
import argparse,json,math
from pathlib import Path
def metrics(rows):
 if len({r['request_id'] for r in rows})!=len(rows):raise ValueError('Duplicate request IDs')
 ranks=[]
 for r in rows:
  target=r['output'].strip(' \n"');pred=[s.strip('"\n').strip() for s in r['predict']]
  ranks.append(pred.index(target)+1 if target in pred else float('inf'))
 return {'denominator':len(rows),'metrics':{key:value for k in (1,5,10,50) for key,value in [(f'HR@{k}',sum(x<=k for x in ranks)/len(rows) if rows else None),(f'NDCG@{k}',sum(1/math.log2(x+1) for x in ranks if x<=k)/len(rows) if rows else None)]}}
def main():
 p=argparse.ArgumentParser();p.add_argument('--predictions',type=Path);a=p.parse_args()
 if a.predictions:print(json.dumps(metrics(json.loads(a.predictions.read_text())),indent=2))
 else:
  root=Path(__file__).resolve().parents[1]
  print((root/'results/catalog_metrics.json').read_text())
if __name__=='__main__':main()
