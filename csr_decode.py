"""CSR constrained decoding for token paths; inspired by STATIC.

Each instance belongs to exactly one generate call and one catalog. Do not share
it across requests or threads. Paths include the response suffix and a final EOS.
"""
import torch
from src.decoding.tensor_index import CSRIndex
from src.decoding.tensor_constraints import TensorConstraints

class CSRLogitsProcessor:
    def __init__(self, paths, eos_token_id, device='cpu'):
        self.constraint=TensorConstraints(CSRIndex(paths,eos=eos_token_id),device=device)
        self.eos=eos_token_id
        self.count=0

    def __call__(self,input_ids,scores):
        norm=torch.nn.functional.log_softmax(scores,dim=-1)
        prefix=input_ids[:,-self.count:] if self.count else input_ids[:,:0]
        lengths=torch.full((len(input_ids),),self.count,dtype=torch.long,device=input_ids.device)
        result,status=self.constraint.mask_scores(prefix,lengths,norm)
        ended=status!=0
        result[ended,self.eos]=norm[ended,self.eos]
        self.count+=1
        return result
