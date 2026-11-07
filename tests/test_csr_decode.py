import unittest
import torch
from csr_decode import CSRLogitsProcessor

class CSRDecodeTests(unittest.TestCase):
    def test_next_tokens_and_catalog_removal(self):
        paths=[[1,3,2],[1,4,2],[5,6,2]]
        for allowed,expected in [(paths,{3,4}),([paths[0],paths[2]],{3})]:
            p=CSRLogitsProcessor(allowed,2)
            scores=torch.arange(8,dtype=torch.float32).repeat(2,1)
            a=p(torch.tensor([[7],[7]]),scores)
            self.assertEqual(set(torch.where(torch.isfinite(a[0]))[0].tolist()),{1,5})
            b=p(torch.tensor([[7,1],[7,5]]),scores)
            self.assertEqual(set(torch.where(torch.isfinite(b[0]))[0].tolist()),expected)
            self.assertEqual(set(torch.where(torch.isfinite(b[1]))[0].tolist()),{6})
            torch.testing.assert_close(b[0,3],torch.log_softmax(scores,dim=-1)[0,3],rtol=0,atol=0)
    def test_eos_and_invalid_prefix(self):
        p=CSRLogitsProcessor([[1,3,2]],2)
        s=torch.zeros(2,8)
        p(torch.tensor([[7],[7]]),s)
        bad=p(torch.tensor([[7,1],[7,6]]),s)
        self.assertEqual(set(torch.where(torch.isfinite(bad[1]))[0].tolist()),{2})
        end=p(torch.tensor([[7,1,3],[7,6,2]]),s)
        self.assertEqual(set(torch.where(torch.isfinite(end[0]))[0].tolist()),{2})
if __name__=='__main__':unittest.main()
