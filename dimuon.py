"""PyTorch implementation of the recovered Dimuon update.

Each parameter is one LoRA factor: [m, n] or [experts, m, n]. Store fused
projections as one parameter. Defaults: FP16 Polar Express, BF16 momentum.
"""
from __future__ import annotations

import math

import torch

_RAW = (
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
)
PE_COEFFICIENTS = tuple((a / 1.01, b / 1.01**3, c / 1.01**5) for a, b, c in _RAW)
_DTYPES = {"fp16": torch.float16, "fp32": torch.float32, "fp64": torch.float64}


def _polar_express(z: torch.Tensor, precision: str) -> torch.Tensor:
    """z: [E, m, n] in fp32/fp64. Returns the 5-step Polar Express output in z's dtype."""
    transpose = z.shape[-2] > z.shape[-1]
    x = z.mT if transpose else z
    norm = torch.linalg.vector_norm(x, dim=(-2, -1), keepdim=True)
    work = _DTYPES[precision]
    guard = 1e-7 if precision == "fp16" else 0.0
    x = (x / (norm * 1.01 + guard)).to(work)
    for a, b, c in PE_COEFFICIENTS:
        gram = x @ x.mT
        middle = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        x = torch.baddbmm(x, middle, x, beta=a, alpha=1.0)
    x = x.to(z.dtype)
    return x.mT if transpose else x


def _conditioner(m_old: torch.Tensor, dim: int, damping: float) -> torch.Tensor:
    energy = (m_old * m_old).sum(dim=dim)                      # [E, k]
    peak = energy.amax(dim=-1, keepdim=True)
    cond = (energy / peak.clamp_min(torch.finfo(energy.dtype).tiny) + damping) ** -0.25
    return torch.where(peak > 0, cond, torch.ones_like(cond))  # identity when the old momentum is zero


class Dimuon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, beta1: float = 0.9, grad_clip_norm: float = 0.0,
                 precision: str = "fp16", damping: float = 1e-4, state_dtype: torch.dtype = torch.float32,
                 momentum_dtype: torch.dtype | None = torch.bfloat16):
        """state_dtype: arithmetic dtype for momentum, conditioners and normalisation.
        momentum_dtype: dtype the momentum is KEPT in between steps (bf16 matches the server; None = state_dtype).
        The new momentum is computed in state_dtype and used unrounded for the current step; only the stored
        copy is rounded."""
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("beta1 must be in [0, 1)")
        if precision not in _DTYPES:
            raise ValueError("precision must be fp16, fp32 or fp64")
        super().__init__(params, {"lr": lr, "beta1": beta1})
        self.grad_clip_norm, self.precision, self.damping, self.state_dtype = grad_clip_norm, precision, damping, state_dtype
        self.momentum_dtype = momentum_dtype or state_dtype

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        params = [p for g in self.param_groups for p in g["params"]]
        grads = [p.grad for p in params if p.grad is not None]
        if not grads and not self.state:
            return loss
        total = torch.sqrt(sum((g.to(self.state_dtype) ** 2).sum() for g in grads)) if grads else None
        scale = 1.0
        if self.grad_clip_norm > 0 and total is not None and float(total) > 0:
            scale = min(1.0, self.grad_clip_norm / float(total))
        for group in self.param_groups:
            lr, beta1 = group["lr"], group["beta1"]
            for p in group["params"]:
                state = self.state[p]
                if p.grad is None and "momentum" not in state:
                    continue
                g = torch.zeros_like(p, dtype=self.state_dtype) if p.grad is None else p.grad.to(self.state_dtype)
                g3 = g if g.dim() == 3 else g.unsqueeze(0)
                m_old = state.get("momentum")
                m_old = torch.zeros_like(g3) if m_old is None else m_old.to(self.state_dtype)
                left = _conditioner(m_old, -1, self.damping)    # [E, m]
                right = _conditioner(m_old, -2, self.damping)   # [E, n]
                m_new = beta1 * m_old + (1.0 - beta1) * scale * g3
                if self.momentum_dtype == torch.bfloat16 and m_new.dtype == torch.float64:
                    stored = m_new.to(torch.float32).to(torch.bfloat16)   # fp32 arithmetic, then one rounding
                else:
                    stored = m_new.to(self.momentum_dtype)
                state["momentum"] = stored
                inner = left[..., :, None] * m_new * right[..., None, :]
                live = torch.linalg.vector_norm(inner, dim=(-2, -1)) > 0
                if not bool(live.any()):
                    continue
                d = left[..., :, None] * _polar_express(inner, self.precision) * right[..., None, :]
                rank = min(d.shape[-2], d.shape[-1])
                target = lr * math.sqrt(rank) * (rank / 32) ** 0.175
                dn = torch.linalg.vector_norm(d, dim=(-2, -1), keepdim=True)
                delta = torch.where(live[:, None, None], -target * d / dn.clamp_min(torch.finfo(d.dtype).tiny), torch.zeros_like(d))
                p.add_(delta.reshape(p.shape).to(p.dtype))
        return loss
