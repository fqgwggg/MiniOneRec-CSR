"""CPU-only runnable example: same prefix, different catalog."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from csr_decode import CSRLogitsProcessor
def main():
 paths=[[1,3,2],[1,4,2],[5,6,2]]
 for name,catalog in [('full',paths),('removed_second_item',[paths[0],paths[2]])]:
  proc=CSRLogitsProcessor(catalog,eos_token_id=2)
  scores=torch.zeros(1,8)
  proc(torch.tensor([[7]]),scores)
  masked=proc(torch.tensor([[7,1]]),scores)
  print(name,torch.where(torch.isfinite(masked[0]))[0].tolist())
if __name__=='__main__':main()
