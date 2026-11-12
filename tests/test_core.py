import unittest
import torch
from speed_limit.cache import validate_cache, validate_tensor_file
from speed_limit.lanczos import lanczos_top_eigenpairs, gradient_gram, adam_preconditioner
from speed_limit.metrics import absorption_curve, absorption_times, aggregate_benchmark_accuracy
from speed_limit.phase0 import top64_kl, distribution
from speed_limit.legacy import check_objective
from speed_limit.queue import resolve_gpu_index

class CoreTests(unittest.TestCase):
    def test_gpu_uuid_resolves_to_numeric_index_for_vllm(self):
        inventory='0, GPU-first\n1, GPU-second\n'
        self.assertEqual(resolve_gpu_index('GPU-second',inventory),'1')
        with self.assertRaises(ValueError): resolve_gpu_index('GPU-absent',inventory)
        with self.assertRaises(ValueError): resolve_gpu_index('GPU-second',inventory+'2, GPU-second\n')

    def test_absorption_censoring_and_actual_steps(self):
        curve=absorption_curve(torch.tensor([[4.,2.],[2.,1.],[.2,.1]]))
        result=absorption_times(curve,[0,10,20])
        self.assertEqual(result['tau_50'],20); self.assertEqual(result['tau_90'],20)
        self.assertIsNone(absorption_times([1.,.7])['tau_90'])
        self.assertTrue(absorption_times([1.,.7])['tau_90_censored'])

    def test_macro(self):
        records=[{'benchmark':'a','correct':True},{'benchmark':'b','correct':False},{'benchmark':'b','correct':False}]
        self.assertEqual(aggregate_benchmark_accuracy(records)['macro_accuracy'],.5)

    def test_cache_reject_alignment_and_missing_candidates(self):
        row={'schema_version':2,'row_id':1,'prompt':[1,2],'response':[3],
             'teacher_logprobs':torch.tensor([-.1]),'top64_ids':torch.arange(64)[None,:],
             'top64_logprobs':torch.full((1,64),-5.),'finish_reason':'stop','initial_student':True}
        self.assertEqual(validate_cache([row])['rows'],1)
        with self.assertRaises(ValueError): validate_cache([row,row])
        with self.assertRaises(ValueError): validate_cache([{**row,'top64_ids':torch.empty(0)}])
        with self.assertRaises(ValueError): validate_cache([{**row,'teacher_logprobs':torch.tensor([float('nan')])}])

    def test_serialized_cache_probability_alignment(self):
        import tempfile
        from pathlib import Path
        logp=torch.randn(2,80).log_softmax(-1); top,ids=logp.topk(64,dim=-1)
        response=ids[:,3]
        row={'schema_version':2,'row_id':7,'prompt':[1,2],'response':response.tolist(),
             'teacher_logprobs':logp.gather(-1,response[:,None]).flatten(),
             'top64_ids':ids.int(),'top64_logprobs':top,'finish_reason':'length','initial_student':True}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'batch.pt'; torch.save([row],path)
            self.assertEqual(validate_tensor_file(path,1)['tokens'],2)
            row['teacher_logprobs']=row['teacher_logprobs']-.5; torch.save([row],path)
            with self.assertRaises(ValueError): validate_tensor_file(path,1)

    def test_kl_gradient_and_support_normalization(self):
        a=torch.tensor([[2.,0.,-1.]],requires_grad=True); b=torch.tensor([[0.,1.,-2.]])
        kl=top64_kl(a,b); self.assertGreater(float(kl),0)
        kl.sum().backward(); self.assertGreater(float(a.grad.norm()),0)
        self.assertTrue(torch.allclose(top64_kl(a,a.detach()),torch.zeros(1),atol=1e-6))
        self.assertTrue(torch.allclose(top64_kl(a+100,b-50),kl,atol=2e-6))
        self.assertLess(check_objective(torch)['gradient_max_error'],1e-10)

    def test_response_alignment_and_live_gradient(self):
        from types import SimpleNamespace
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight=torch.nn.Parameter(torch.randn(70,70))
            def forward(self,input_ids,use_cache=False):
                return SimpleNamespace(logits=self.weight[input_ids])
        model=Tiny(); row={'prompt':[2,4,6],'response':[8,10]}
        sampled,ids,top=distribution(model,row,teacher=True)
        expected=model.weight[[6,8]].log_softmax(-1).gather(-1,torch.tensor([[8],[10]])).flatten()
        self.assertTrue(torch.allclose(sampled,expected,atol=1e-6))
        row.update(top64_ids=ids)
        current,candidates=distribution(model,row,gradients=True)
        current.sum().backward()
        self.assertGreater(float(model.weight.grad.norm()),0)
        self.assertEqual(candidates.shape,(2,64))

    def test_matrixfree_kernel_matches_explicit_gradient_gram(self):
        torch.manual_seed(17)
        x=torch.randn(7,4,dtype=torch.float64); target=torch.randn(7,dtype=torch.float64)
        params={'w':torch.randn(4,dtype=torch.float64)}
        def losses(p): return .5*(x@p['w']-target).square()
        G=torch.func.jacrev(losses)(params)['w']; explicit=G@G.T
        _,operator=gradient_gram(losses,params)
        u=torch.randn(7,dtype=torch.float64)
        self.assertTrue(torch.allclose(operator(u),explicit@u,atol=1e-9))
        values,_=lanczos_top_eigenpairs(operator,7,7,dtype=torch.float64)
        reference=torch.linalg.eigvalsh(explicit).flip(0)[:len(values)]
        relative=float((values-reference).abs().max()/reference[0])
        self.assertLess(relative,.01)
        v={'w':torch.arange(1,5,dtype=torch.float64)}
        P=adam_preconditioner(params,v,step=3)
        _,preconditioned=gradient_gram(losses,params,P)
        self.assertTrue(torch.allclose(preconditioned(u),(G*P['w'])@G.T@u,atol=1e-6))

    def test_lanczos_early_breakdown(self):
        values,vectors=lanczos_top_eigenpairs(lambda x:x*3,5,5,dtype=torch.float64)
        self.assertEqual(len(values),1); self.assertAlmostEqual(float(values[0]),3)

if __name__=='__main__': unittest.main()
