"""Real model.generate A/B; no prediction cache. User supplies model and tokens.
Input JSON: {"eos_token_id":2,"pad_token_id":2,"paths":[[1,3,2]],
"input_ids":[[7,8]],"attention_mask":[[1,1]]}.
This benchmark uses the packaged dictionary oracle, not the original paper kernel.
"""
import argparse,hashlib,json,statistics,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from csr_decode import CSRLogitsProcessor
from src.decoding.reference_constraints import ReferenceConstraints
class ReferenceProcessor(CSRLogitsProcessor):
 def __init__(self,paths,eos_token_id,device='cpu'):
  self.constraint=ReferenceConstraints(paths,eos=eos_token_id);self.eos=eos_token_id;self.count=0
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--model',required=True);p.add_argument('--revision');p.add_argument('--inputs',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda')
 p.add_argument('--dtype',choices=['float32','bfloat16','float16'],default='bfloat16')
 p.add_argument('--beams',type=int,default=50);p.add_argument('--repeats',type=int,default=5)
 a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 if a.repeats<1 or a.beams<1:raise ValueError('Positive repeats and beams required')
 from transformers import AutoModelForCausalLM,GenerationConfig,LogitsProcessorList
 data=json.loads(a.inputs.read_text());device=torch.device(a.device)
 x=torch.tensor(data['input_ids'],dtype=torch.long,device=device)
 attention=torch.tensor(data['attention_mask'],dtype=torch.long,device=device)
 assert x.shape==attention.shape
 model=AutoModelForCausalLM.from_pretrained(a.model,revision=a.revision,torch_dtype=getattr(torch,a.dtype)).to(device).eval()
 config=GenerationConfig(num_beams=a.beams,num_return_sequences=a.beams,do_sample=False,
  max_new_tokens=max(map(len,data['paths'])),eos_token_id=data['eos_token_id'],pad_token_id=data['pad_token_id'],length_penalty=0.0)
 def sync():
  if device.type=='cuda':torch.cuda.synchronize(device)
 def generate(cls):
  sync();start=time.perf_counter();processor=cls(data['paths'],data['eos_token_id'],device=device);sync();build=time.perf_counter()-start
  sync();start=time.perf_counter()
  with torch.inference_mode():out=model.generate(input_ids=x,attention_mask=attention,generation_config=config,do_sample=False,logits_processor=LogitsProcessorList([processor]))
  sync();elapsed=time.perf_counter()-start
  return out,build,elapsed
 times={'reference':[],'csr':[]};builds={'reference':[],'csr':[]};classes={'reference':ReferenceProcessor,'csr':CSRLogitsProcessor}
 for cls in classes.values():generate(cls)
 for i in range(a.repeats):
  outputs={}
  for name in (['reference','csr'] if i%2==0 else ['csr','reference']):
   out,build,elapsed=generate(classes[name]);outputs[name]=out;times[name].append(elapsed);builds[name].append(build)
  if not torch.equal(outputs['reference'],outputs['csr']):raise RuntimeError('Output parity failed; no speedup claim')
 med={n:statistics.median(v) for n,v in times.items()}
 report={'model':a.model,'revision':a.revision,'input_sha256':hashlib.sha256(a.inputs.read_bytes()).hexdigest(),
  'torch':torch.__version__,'device':str(device),'dtype':a.dtype,'beams':a.beams,'requests':len(x),
  'seconds':times,'index_build_seconds':builds,'median_seconds':med,'tokens_equal':True,
  'speedup':med['reference']/med['csr'],'reduction':1-med['csr']/med['reference'],
  'scope':'caller-supplied workload; complete generation timing excludes index construction, model loading and tokenization'}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
