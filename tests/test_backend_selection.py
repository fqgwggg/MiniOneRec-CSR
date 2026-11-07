import unittest,sys,types
from unittest.mock import patch
import torch
from constraint_backend import PreparedCSR,make_processor,validate_backend
class BackendTests(unittest.TestCase):
 def test_independent_calls_and_header(self):
  p=PreparedCSR([[7,8,1,3,2],[7,8,1,4,2]],2,2,'cpu')
  a=make_processor('csr',prepared=p);b=make_processor('csr',prepared=p)
  scores=torch.randn(2,10)
  x=torch.tensor([[0,7,8],[0,7,8]])
  self.assertTrue(torch.equal(a(x,scores),b(x,scores)))
  a(torch.tensor([[7,8,1],[7,8,1]]),scores)
  self.assertEqual(b.count,1)
  self.assertIs(a.constraint,b.constraint)
  fresh=make_processor('csr',prepared=p)
  bad=fresh(torch.tensor([[0,7,8],[0,6,8]]),scores)
  self.assertEqual(torch.where(torch.isfinite(bad[1]))[0].tolist(),[2])
 def test_invalid_backend(self):
  with self.assertRaises(ValueError):validate_backend('typo')
 def test_reference_dispatch_unchanged(self):
  fake=types.ModuleType('LogitProcessor');fake.ConstrainedLogitsProcessor=lambda **kw:kw
  with patch.dict(sys.modules,{'LogitProcessor':fake}):
   self.assertEqual(make_processor('reference',num_beams=16,eos_token_id=2),{'num_beams':16,'eos_token_id':2})
 def test_catalog_rebuild(self):
  p=PreparedCSR([[7,8,1,3,2]],2,2,'cpu');a=make_processor('csr',prepared=p)
  a(torch.tensor([[7,8]]),torch.zeros(1,10))
  r=a(torch.tensor([[7,8,1]]),torch.zeros(1,10))
  self.assertEqual(torch.where(torch.isfinite(r[0]))[0].tolist(),[3])
if __name__=='__main__':unittest.main()
