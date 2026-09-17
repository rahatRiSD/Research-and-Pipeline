"""Objectives.

ARBITER total loss
------------------
    L = CE(z1) + CE(z2) + CE(z3)            per-level task loss (keeps every
                                            rung trainable even when the router
                                            saturates)
      + CE(z_soft)                          end-to-end loss through soft routing
      + w_r * sum_k BCE(r_k, 1[level k correct])     utility supervision
      + w_m * marginal-order regulariser    encourages r1 <= r2 <= r3 only where
                                            the deeper level actually helps
      + w_e * EDL(z3)                       evidential calibration on the deep head
      + w_b * E[cost]                       budget penalty (differentiable)

The router term is the core novelty: instead of one binary "is stream A right",
each rung is supervised against its own realised correctness, which makes the
difference r_{k+1} - r_k a direct estimate of the marginal gain of escalating.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Evidential deep learning (Sensoy et al., 2018)
# --------------------------------------------------------------------------- #
def edl_mse_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int,
                 epoch: int = 0, anneal_epochs: int = 10) -> torch.Tensor:
    evidence = F.softplus(logits)
    alpha = evidence + 1.0
    S = alpha.sum(dim=1, keepdim=True)
    y = F.one_hot(target, num_classes).float()
    p = alpha / S
    err = ((y - p) ** 2).sum(dim=1)
    var = (p * (1 - p) / (S + 1.0)).sum(dim=1)
    loss = err + var

    # KL to the uniform Dirichlet on the mis-evidence, annealed in
    alpha_tilde = y + (1 - y) * alpha
    kl = _kl_dirichlet(alpha_tilde, num_classes)
    coef = min(1.0, float(epoch) / max(anneal_epochs, 1))
    return (loss + coef * kl).mean()


def _kl_dirichlet(alpha: torch.Tensor, num_classes: int) -> torch.Tensor:
    ones = torch.ones_like(alpha)
    S = alpha.sum(dim=1, keepdim=True)
    t1 = torch.lgamma(S).squeeze(1) - torch.lgamma(alpha).sum(dim=1)
    t2 = torch.lgamma(ones).sum(dim=1) - torch.lgamma(ones.sum(dim=1))
    t3 = ((alpha - ones) * (torch.digamma(alpha) - torch.digamma(S))).sum(dim=1)
    return t1 + t2 + t3


# --------------------------------------------------------------------------- #
class FocalLoss(nn.Module):
    """Useful for FracAtlas, where fractures are ~18% of the scans."""

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# --------------------------------------------------------------------------- #
@dataclass
class ArbiterLossConfig:
    w_level: float = 1.0
    w_soft: float = 1.0
    w_router: float = 1.0
    w_order: float = 0.1
    w_evid: float = 0.2
    w_budget: float = 0.05
    label_smoothing: float = 0.05
    focal_gamma: float = 0.0
    anneal_epochs: int = 10


class ArbiterLoss(nn.Module):
    def __init__(self, num_classes: int, cfg: ArbiterLossConfig, class_weight: Optional[torch.Tensor] = None,
                 level_costs=(0.0, 0.03, 1.0)):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.level_costs = level_costs
        self.register_buffer("class_weight", class_weight if class_weight is not None else torch.ones(num_classes))
        self.focal = FocalLoss(cfg.focal_gamma, class_weight) if cfg.focal_gamma > 0 else None

    def _task(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.focal is not None:
            return self.focal(logits, y)
        return F.cross_entropy(logits, y, weight=self.class_weight.to(logits.device),
                               label_smoothing=self.cfg.label_smoothing)

    def forward(self, out: Dict[str, torch.Tensor], y: torch.Tensor, epoch: int = 0) -> Dict[str, torch.Tensor]:
        c = self.cfg
        l1, l2, l3 = out["z1"], out["z2"], out["z3"]

        task = c.w_level * (self._task(l1, y) + self._task(l2, y) + self._task(l3, y))
        soft = c.w_soft * self._task(out["z_soft"], y)

        # --- router utility supervision --------------------------------- #
        with torch.no_grad():
            correct = torch.stack([
                (l1.argmax(1) == y).float(),
                (l2.argmax(1) == y).float(),
                (l3.argmax(1) == y).float(),
            ], dim=1)
        router = c.w_router * F.binary_cross_entropy_with_logits(out["router_logits"], correct)

        # --- monotone-utility regulariser -------------------------------- #
        r = out["router_probs"]
        gain21 = (correct[:, 1] - correct[:, 0])
        gain32 = (correct[:, 2] - correct[:, 1])
        pred21 = r[:, 1] - r[:, 0]
        pred32 = r[:, 2] - r[:, 1]
        order = c.w_order * (F.smooth_l1_loss(pred21, gain21) + F.smooth_l1_loss(pred32, gain32))

        # --- evidential calibration on the deep head --------------------- #
        evid = c.w_evid * edl_mse_loss(l3, y, self.num_classes, epoch, c.anneal_epochs) if c.w_evid > 0 else l3.sum() * 0

        # --- expected compute --------------------------------------------- #
        cost = (out["gate2"] * (self.level_costs[1] - self.level_costs[0])
                + out["gate3"] * (self.level_costs[2] - self.level_costs[1])).mean()
        budget = c.w_budget * cost

        total = task + soft + router + order + evid + budget
        return {"loss": total, "task": task.detach(), "soft": soft.detach(),
                "router": router.detach(), "order": order.detach(),
                "evid": evid.detach() if torch.is_tensor(evid) else torch.tensor(0.0),
                "budget": budget.detach(), "expected_cost": cost.detach()}


# --------------------------------------------------------------------------- #
class SecondOpinionLoss(nn.Module):
    """Four-term objective of the prior SOTA (reimplemented for fair comparison)."""

    def __init__(self, class_weight: Optional[torch.Tensor] = None, label_smoothing: float = 0.0):
        super().__init__()
        self.register_buffer("class_weight", class_weight if class_weight is not None else None,
                             persistent=False)
        self.ls = label_smoothing

    def _ce(self, logits, y):
        w = self.class_weight.to(logits.device) if self.class_weight is not None else None
        return F.cross_entropy(logits, y, weight=w, label_smoothing=self.ls)

    def forward(self, out: Dict[str, torch.Tensor], y: torch.Tensor, epoch: int = 0) -> Dict[str, torch.Tensor]:
        za, zf, alpha = out["z_a"], out["z_fused"], out["alpha"]
        with torch.no_grad():
            g = (za.argmax(1) == y).float()
        loss = self._ce(out["logits"], y) + self._ce(za, y) + self._ce(zf, y) \
            + F.binary_cross_entropy(alpha.clamp(1e-6, 1 - 1e-6), g)
        return {"loss": loss, "task": loss.detach()}


class PlainLoss(nn.Module):
    def __init__(self, class_weight: Optional[torch.Tensor] = None, label_smoothing: float = 0.05,
                 focal_gamma: float = 0.0):
        super().__init__()
        self.register_buffer("class_weight", class_weight if class_weight is not None else None,
                             persistent=False)
        self.ls = label_smoothing
        self.focal = FocalLoss(focal_gamma, class_weight) if focal_gamma > 0 else None

    def forward(self, out: Dict[str, torch.Tensor], y: torch.Tensor, epoch: int = 0) -> Dict[str, torch.Tensor]:
        if self.focal is not None:
            loss = self.focal(out["logits"], y)
        else:
            w = self.class_weight.to(out["logits"].device) if self.class_weight is not None else None
            loss = F.cross_entropy(out["logits"], y, weight=w, label_smoothing=self.ls)
        return {"loss": loss, "task": loss.detach()}


def build_loss(model_name: str, num_classes: int, class_weight: Optional[torch.Tensor] = None,
               cfg: Optional[dict] = None) -> nn.Module:
    cfg = cfg or {}
    if model_name == "arbiter":
        lcfg = ArbiterLossConfig(**{k: v for k, v in cfg.items() if k in ArbiterLossConfig.__annotations__})
        return ArbiterLoss(num_classes, lcfg, class_weight, cfg.get("level_costs", (0.0, 0.03, 1.0)))
    if model_name == "second_opinion":
        return SecondOpinionLoss(class_weight, cfg.get("label_smoothing", 0.0))
    return PlainLoss(class_weight, cfg.get("label_smoothing", 0.05), cfg.get("focal_gamma", 0.0))
