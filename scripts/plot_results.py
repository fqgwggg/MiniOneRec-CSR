"""Render figures from saved numerical evidence; no model or GPU required."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def main():
 r=ROOT/'results';out=ROOT/'assets/results';out.mkdir(parents=True,exist_ok=True)
 d=json.loads((r/'catalog_metrics.json').read_text());fig,axes=plt.subplots(1,3,figsize=(12,3.8))
 for ax,model in zip(axes,['sft','rl','sasrec']):
  rows=d[model];x=list(range(len(rows)));labels=[row['catalog'].replace('seed','').replace('_keep',' / ') for row in rows]
  ax.plot(x,[row['paired_full_catalog']['metrics']['HR@10'] for row in rows],'o--',label='Full, same cohort')
  ax.plot(x,[row['paired_dynamic']['metrics']['HR@10'] for row in rows],'o-',label='Subset, same cohort')
  ax.set_xticks(x,labels,rotation=70,fontsize=7);ax.set_title(model.upper());ax.set_ylabel('HR@10');ax.legend(fontsize=7)
 fig.suptitle('Available-target paired cohorts; different snapshots have different denominators');fig.tight_layout();fig.savefig(out/'catalog.png',dpi=160);plt.close(fig)
 a=json.loads((r/'generation_benchmark.json').read_text());b=json.loads((r/'rl_update_benchmark.json').read_text());c=json.loads((r/'rl_update_benchmark_reverse.json').read_text())
 groups=[
  ('Constraint callback',[1000*a['instrumented_constraint_seconds'][n] for n in ['reference','static']],'Profiled time (ms)','GPU-synchronized module timing'),
  ('Full generation (beam50)',[a['hot_generation_seconds'][n] for n in ['reference','static']],'Generation time (s)','Complete-generation timing'),
  ('Full RL update',[b['timing'][n]['seconds_per_update_mean'] for n in ['reference','static']],'Mean time per update (s)','10 measured updates after warmup')]
 fig,axes=plt.subplots(1,3,figsize=(12.8,4.4))
 for ax,(title,values,ylabel,scope) in zip(axes,groups):
  bars=ax.bar(['Reference','CSR'],values,color=['#9aa8b3','#2878b5'],width=0.55)
  ax.set_ylim(0,max(values)*1.3);ax.set_ylabel(ylabel);ax.bar_label(bars,fmt='%.3f',padding=4)
  reduction=100*(1-values[1]/values[0]);ax.set_title(title+'\n'+f'{reduction:.2f}% less time',fontsize=12)
  ax.text(0.5,-0.18,scope,ha='center',va='top',transform=ax.transAxes,fontsize=8.5)
  ax.spines[['top','right']].set_visible(False)
 fig.suptitle('Reference vs CSR constrained decoding',fontsize=15)
 fig.text(0.5,0.015,'Independent timing scopes and axes. RL reverse-order repeat: '+f"{100*c['reduction_fraction']:.2f}% reduction. Constraints remain enabled in both backends.",ha='center',fontsize=9)
 fig.tight_layout(rect=[0,0.09,1,0.94]);fig.savefig(out/'performance.png',dpi=180);plt.close(fig)
if __name__=='__main__':main()
