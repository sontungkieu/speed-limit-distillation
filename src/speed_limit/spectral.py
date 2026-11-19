"""Rank-16 LoRA and a streamed, per-token Gram operator for K3."""
import math

import torch
from torch import nn
from torch.func import functional_call, jvp
from torch.nn import functional as F

from .phase0 import top64_kl


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.a = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + self.scale * F.linear(F.linear(x, self.a), self.b)


def enable_lora(student, rank=16, alpha=32):
    student.requires_grad_(False)
    paths = [name for name, module in student.model.named_modules()
             if isinstance(module, nn.Linear)]
    if not paths:
        raise ValueError('No transformer linear layers found')
    for path in paths:
        parent, _, attribute = path.rpartition('.')
        owner = student.model.get_submodule(parent)
        layer = getattr(owner, attribute)
        setattr(owner, attribute, LoRALinear(layer, rank, alpha))
    return paths


class CandidateLoss(nn.Module):
    """Qwen3 reverse KL on the cached teacher support at each response state."""

    def __init__(self, student):
        super().__init__()
        self.student = student

    def forward(self, batch):
        tokens, start, index, teacher = batch
        states = self.student.model(input_ids=tokens, use_cache=False).last_hidden_state
        states = states[0, start:start + index.shape[0]]
        weight = self.student.lm_head.weight
        losses = []
        for offset in range(0, index.shape[0], 128):
            chunk = slice(offset, offset + 128)
            candidates = weight[index[chunk]]
            logits = torch.bmm(candidates, states[chunk].unsqueeze(-1)).squeeze(-1)
            losses.append(top64_kl(logits, teacher[chunk]))
        return torch.cat(losses)


class StreamedGram:
    """Compute G P Gᵀ with one rollout graph at a time. G includes 1/√m."""

    def __init__(self, model, batches, preconditioner=None):
        self.model = model
        trainable = {name: value for name, value in model.named_parameters() if value.requires_grad}
        if not batches or not trainable:
            raise ValueError('Gram requires states and trainable parameters')
        self.names = tuple(trainable)
        self.parameters = tuple(trainable.values())
        self.batches = batches
        self.sizes = [batch[2].shape[0] for batch in batches]
        self.preconditioner = preconditioner

    def losses(self, values, batch):
        return functional_call(self.model, dict(zip(self.names, values)), (batch,))

    def matvec(self, u):
        if u.shape != (sum(self.sizes),):
            raise ValueError('Gram vector shape differs from token-state count')
        tangent = [torch.zeros_like(value) for value in self.parameters]
        for batch, coefficients in zip(self.batches, u.split(self.sizes)):
            losses = self.losses(self.parameters, batch)
            gradients = torch.autograd.grad(losses @ coefficients, self.parameters)
            for value, gradient in zip(tangent, gradients):
                value.add_(gradient)
        if self.preconditioner is not None:
            tangent = [value * self.preconditioner[name] for name, value in zip(self.names, tangent)]
        with torch.no_grad():
            result = [jvp(lambda *values: self.losses(values, batch), self.parameters,
                          tuple(tangent))[1] for batch in self.batches]
        return torch.cat(result) / len(self.batches)


def residual_prediction(losses, eigenvalues, eigenvectors, lr, steps):
    """Record the sqrt(2 KL) proxy and hold unmeasured residual mass constant."""
    residual = (2 * losses.clamp_min(0)).sqrt().double()
    mass = residual.square().sum()
    if not torch.isfinite(mass) or mass <= 0:
        raise ValueError('Residual proxy has no finite positive mass')
    fractions = (eigenvectors.double().T @ residual).square() / mass
    unresolved = (1. - fractions.sum()).clamp(0., 1.)
    eigenvalues = eigenvalues.double().clamp_min(0.)
    rates = (1. - lr * eigenvalues).square()
    if (rates > 1.).any():
        return fractions, unresolved, None
    exponent = torch.arange(steps + 1, device=rates.device)
    curve = rates[None, :].pow(exponent[:, None]) @ fractions + unresolved
    return fractions, unresolved, curve
