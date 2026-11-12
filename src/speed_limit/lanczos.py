"""Matrix-free VJP/JVP loss-gradient Gram operator and symmetric Lanczos."""
import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

def gradient_gram(losses,parameters,preconditioner=None):
    """losses(parameters) -> per-state vector. G is never materialized.

    Detached advantages must be frozen outside losses, otherwise the JVP would
    differentiate through a stop-gradient differently from the intended G.
    Parameters and optimizer second moments are FP32 for real-model diagnostics.
    """
    values,pullback=torch.func.vjp(losses,parameters)
    def matvec(u):
        tangent=pullback(u.to(values))[0]
        if preconditioner is not None:
            leaves,spec=tree_flatten(tangent); weights,_=tree_flatten(preconditioner)
            tangent=tree_unflatten([a*b for a,b in zip(leaves,weights)],spec)
        return torch.func.jvp(losses,(parameters,),(tangent,))[1]
    return values,matvec

def adam_preconditioner(parameters,second_moment,step,beta2=.999,eps=1e-8):
    if step<=0: raise ValueError('Adam preconditioner requires measured optimizer moments')
    return {name:1/((second_moment[name].float()/(1-beta2**step)).sqrt()+eps) for name in parameters}

def lanczos_top_eigenpairs(matvec,dimension,iterations=50,dtype=torch.float32,device='cpu',seed=0):
    if dimension<=0 or iterations<=0: raise ValueError('positive dimensions required')
    generator=torch.Generator(device=device).manual_seed(seed)
    q=torch.randn(dimension,dtype=dtype,device=device,generator=generator); q=q/q.norm()
    basis=[]; diagonal=[]; off=[]; previous=torch.zeros_like(q); beta=q.new_zeros(())
    for i in range(min(dimension,iterations)):
        basis.append(q); z=matvec(q).to(q)
        if z.shape!=q.shape or not torch.isfinite(z).all(): raise ValueError('invalid matvec')
        z=z-beta*previous; alpha=q@z; z=z-alpha*q
        for _ in range(2):
            for vector in basis: z=z-(vector@z)*vector
        diagonal.append(alpha); next_beta=z.norm()
        if next_beta <= torch.finfo(dtype).eps*max(1.,float(alpha.abs()))*10 or i+1==min(dimension,iterations): break
        off.append(next_beta); previous,q=q,z/next_beta; beta=next_beta
    tri=torch.diag(torch.stack(diagonal))
    if off:
        b=torch.stack(off); tri=tri+torch.diag(b,1)+torch.diag(b,-1)
    values,vectors=torch.linalg.eigh(tri)
    return values.flip(0),(torch.stack(basis,dim=1)@vectors).flip(1)
