"""Backend selection shared by official evaluation and RL entrypoints."""
import torch
from csr_decode import CSRLogitsProcessor
from src.decoding.tensor_index import CSRIndex
from src.decoding.tensor_constraints import TensorConstraints

def validate_backend(backend):
    if backend not in ('reference','csr'):
        raise ValueError('constraint_backend must be reference or csr')

class PreparedCSR:
    """One immutable device index per catalog; new mutable processor per call."""
    def __init__(self, prefix_paths, prefix_index, eos, device):
        if not prefix_paths:raise ValueError('Empty catalog')
        headers={tuple(p[:prefix_index]) for p in prefix_paths}
        if len(headers)!=1 or any(len(p)<=prefix_index for p in prefix_paths):
            raise ValueError('Catalog response headers differ or path is empty')
        self.header=torch.tensor(next(iter(headers)),dtype=torch.long,device=device)
        self.constraint=TensorConstraints(CSRIndex([p[prefix_index:] for p in prefix_paths],eos=eos),device=device)
        self.eos=eos

class PreparedProcessor(CSRLogitsProcessor):
    def __init__(self,prepared):
        self.constraint=prepared.constraint;self.eos=prepared.eos;self.header=prepared.header;self.count=0
    def __call__(self,ids,scores):
        first=self.count==0
        result=super().__call__(ids,scores)
        if first:
            valid=(ids[:,-len(self.header):]==self.header).all(dim=1) if ids.shape[1]>=len(self.header) else torch.zeros(len(ids),dtype=torch.bool,device=ids.device)
            result[~valid]=-torch.inf
            result[~valid,self.eos]=torch.log_softmax(scores,dim=-1)[~valid,self.eos]
        return result

def make_processor(backend, *, prepared=None, **reference_kwargs):
    validate_backend(backend)
    if backend=='reference':
        from LogitProcessor import ConstrainedLogitsProcessor
        return ConstrainedLogitsProcessor(**reference_kwargs)
    if prepared is None:raise ValueError('CSR backend requires a prepared catalog')
    return PreparedProcessor(prepared)
