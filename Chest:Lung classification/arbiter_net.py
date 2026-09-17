"""ARBITER — Anatomy-Routed Budgeted Inference with Trusted Evidential Reasoning.

Motivation and delta over prior work
------------------------------------
Anatomy-guided dual-stream networks (e.g. PelFANet) run both streams on every
input. SecondOpinion improves on that with a binary gate trained as a
correctness classifier, but it still has three limitations that this model
targets:

  (i)  the escalation is *binary* — a case is either cheap or fully dual-stream,
       with nothing in between, even though most uncertain cases need only a hint
       of anatomical context rather than a second full backbone;
  (ii) the gate predicts "is the primary stream correct", which is not the
       quantity the decision actually needs: the decision needs the *expected
       gain* of escalating relative to its cost;
  (iii) the operating point is fixed at 0.5, so the accuracy/compute trade-off
       cannot be moved at deployment time without retraining.

ARBITER addresses all three:

  1. THREE-LEVEL CASCADE instead of a binary gate.
       L1  raw-image backbone                                    (always run)
       L2  FiLM anatomy modulation of the *same* L1 feature map  (+0.9M, +0.01G)
       L3  independent anatomy backbone + cross-attention fusion (+4.5M, +0.41G)
     L2 is the novel middle rung: anatomy is injected as feature-wise affine
     modulation conditioned on a tiny mask encoder, so anatomical context costs
     almost nothing when a full second backbone is overkill.

  2. UTILITY ROUTER. A single head predicts P(level k is correct) for k=1,2,3
     from L1 evidence. At inference we escalate while the *predicted marginal
     gain* exceeds the price of the next rung:
            escalate k -> k+1   iff   r_{k+1} - r_k  >  lambda * (c_{k+1} - c_k)
     The router is supervised per level against the realised correctness of that
     level, so it learns a utility surface rather than a confidence score.

  3. BUDGET KNOB. lambda is a deployment-time scalar, not a trained parameter.
     Sweeping it traces a full accuracy-vs-FLOPs Pareto curve from one set of
     weights (Figure 3), which no fixed-threshold gate can do.

  4. EVIDENTIAL HEAD on the deepest active level. Dirichlet evidence gives a
     calibrated uncertainty u = C/S that feeds back into the router input, so
     routing reacts to epistemic uncertainty (out-of-distribution, hard cases)
     and not only to softmax sharpness.

Everything is differentiable at train time (all levels are evaluated, routing is
soft); hard routing is used only at inference.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import create_backbone


# --------------------------------------------------------------------------- #
# Tiny anatomy encoder
# --------------------------------------------------------------------------- #
class AnatomyEncoder(nn.Module):
    """~0.4M-param encoder over the binary anatomy mask.

    Returns a global anatomy code (for FiLM) and a coarse spatial map (for the
    anatomical attention gate). Deliberately tiny: the whole point of level 2 is
    that anatomical conditioning should not cost a second backbone.
    """

    def __init__(self, in_ch: int = 1, width: int = 32, out_dim: int = 256):
        super().__init__()
        chs = [in_ch, width, width * 2, width * 4, out_dim]
        blocks: List[nn.Module] = []
        for i in range(4):
            blocks += [
                nn.Conv2d(chs[i], chs[i + 1], 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(chs[i + 1]),
                nn.ReLU(inplace=True),
            ]
        self.stem = nn.Sequential(*blocks)          # 224 -> 14
        self.pool = nn.AdaptiveAvgPool2d(7)          # -> 7x7 to match CNN stride 32
        self.out_dim = out_dim

    def forward(self, mask: torch.Tensor):
        m = self.stem(mask)
        m = self.pool(m)
        return m, m.mean(dim=(2, 3))


class FiLM(nn.Module):
    """Feature-wise linear modulation of the primary feature map by anatomy."""

    def __init__(self, code_dim: int, feat_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(code_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * feat_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)  # starts as identity => L2 == L1 at init

    def forward(self, feat: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.net(code).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return feat * (1.0 + torch.tanh(gamma)) + beta


class AnatomySpatialGate(nn.Module):
    """Spatial re-weighting of primary features by the anatomy map."""

    def __init__(self, anat_dim: int, feat_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(anat_dim, feat_dim, kernel_size=1)

    def forward(self, feat: torch.Tensor, anat_map: torch.Tensor) -> torch.Tensor:
        if anat_map.shape[-2:] != feat.shape[-2:]:
            anat_map = F.interpolate(anat_map, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        attn = torch.sigmoid(self.proj(anat_map))
        self.last_attn = attn.detach()
        return feat * (1.0 + attn)


# --------------------------------------------------------------------------- #
# Cross-attention fusion (level 3)
# --------------------------------------------------------------------------- #
class CrossAttentionFusion(nn.Module):
    def __init__(self, dim_q: int, dim_kv: int, dim: int = 256, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.q = nn.Linear(dim_q, dim)
        self.k = nn.Linear(dim_kv, dim)
        self.v = nn.Linear(dim_kv, dim)
        self.out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.last_attn: Optional[torch.Tensor] = None

    def forward(self, fa: torch.Tensor, fb: torch.Tensor) -> torch.Tensor:
        b, ca, h, w = fa.shape
        qa = fa.flatten(2).transpose(1, 2)           # (B, N, Ca)
        kb = fb.flatten(2).transpose(1, 2)           # (B, M, Cb)
        q = self.q(qa).view(b, -1, self.heads, self.dim // self.heads).transpose(1, 2)
        k = self.k(kb).view(b, -1, self.heads, self.dim // self.heads).transpose(1, 2)
        v = self.v(kb).view(b, -1, self.heads, self.dim // self.heads).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (self.dim // self.heads) ** 0.5
        attn = scores.softmax(dim=-1)
        self.last_attn = attn.detach()
        ctx = (self.drop(attn) @ v).transpose(1, 2).reshape(b, -1, self.dim)
        ctx = self.out(ctx)
        return self.norm(ctx.mean(dim=1))


# --------------------------------------------------------------------------- #
# Utility router
# --------------------------------------------------------------------------- #
class UtilityRouter(nn.Module):
    """Predicts P(level k correct) for k = 1..K from level-1 evidence."""

    def __init__(self, feat_dim: int, num_classes: int, anat_dim: int, levels: int = 3, hidden: int = 256):
        super().__init__()
        in_dim = feat_dim + num_classes + 2 + anat_dim  # +entropy, +evidential u
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, levels),
        )
        self.levels = levels
        self.in_dim = in_dim

    @staticmethod
    def build_input(h: torch.Tensor, probs: torch.Tensor, entropy: torch.Tensor,
                    unc: torch.Tensor, anat: torch.Tensor) -> torch.Tensor:
        return torch.cat([h, probs, entropy.unsqueeze(1), unc.unsqueeze(1), anat], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # logits; sigmoid gives predicted correctness per level


# --------------------------------------------------------------------------- #
# Config + model
# --------------------------------------------------------------------------- #
@dataclass
class ArbiterConfig:
    num_classes: int = 2
    backbone: str = "efficientnet_b0"
    pretrained: bool = True
    anat_dim: int = 256
    fusion_dim: int = 256
    fusion_heads: int = 4
    lam: float = 1.0                 # budget knob (higher => cheaper)
    level_costs: List[float] = field(default_factory=lambda: [0.0, 0.03, 1.0])
    dropout: float = 0.2
    evidential: bool = True
    # Ablation switches consumed by scripts/train.py --ablate, used for Table 4:
    #   no_film        remove FiLM anatomy modulation from level 2
    #   no_spatial     remove the anatomical spatial gate
    #   no_l2          collapse the cascade back to a binary (level 1 / level 3) gate
    #   no_evid_input  hide the evidential uncertainty from the router
    #   no_anat_input  hide the anatomy code from the router
    ablate: List[str] = field(default_factory=list)


class ArbiterNet(nn.Module):
    def __init__(self, cfg: ArbiterConfig):
        super().__init__()
        self.cfg = cfg
        C = cfg.num_classes

        # --- Level 1: primary stream (always executed) ----------------------
        self.backbone_a = create_backbone(cfg.backbone, cfg.pretrained, in_chans=3)
        d = self.backbone_a.num_features
        self.head1 = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(d, C))

        # --- Level 2: anatomy modulation of the SAME feature map ------------
        self.anat_enc = AnatomyEncoder(in_ch=1, out_dim=cfg.anat_dim)
        self.film = FiLM(cfg.anat_dim, d)
        self.spatial_gate = AnatomySpatialGate(cfg.anat_dim, d)
        self.head2 = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(d, C))

        # --- Level 3: independent anatomy backbone + cross-attention --------
        self.backbone_b = create_backbone(cfg.backbone, cfg.pretrained, in_chans=3)
        db = self.backbone_b.num_features
        self.fusion = CrossAttentionFusion(d, db, cfg.fusion_dim, cfg.fusion_heads)
        self.head3 = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_dim + d + db, 512),
            nn.GELU(),
            nn.Linear(512, C),
        )

        # --- Router ---------------------------------------------------------
        self.router = UtilityRouter(d, C, cfg.anat_dim, levels=3)

    # ------------------------------------------------------------------ #
    @staticmethod
    def entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return -(probs * (probs + eps).log()).sum(dim=1)

    @staticmethod
    def evidential_uncertainty(logits: torch.Tensor) -> torch.Tensor:
        """Dirichlet vacuity u = C / sum(alpha), alpha = softplus(logits) + 1."""
        alpha = F.softplus(logits) + 1.0
        return logits.shape[1] / alpha.sum(dim=1)

    # ------------------------------------------------------------------ #
    def forward_level1(self, x_a: torch.Tensor) -> Dict[str, torch.Tensor]:
        fa, ha = self.backbone_a(x_a)
        z1 = self.head1(ha)
        return {"fa": fa, "ha": ha, "z1": z1}

    def forward_level2(self, state: Dict[str, torch.Tensor], mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        anat_map, anat_code = self.anat_enc(mask)
        f2 = state["fa"]
        if "no_spatial" not in self.cfg.ablate:
            f2 = self.spatial_gate(f2, anat_map)
        if "no_film" not in self.cfg.ablate:
            f2 = self.film(f2, anat_code)
        h2 = f2.mean(dim=(2, 3))
        state.update({"anat_map": anat_map, "anat_code": anat_code, "f2": f2, "h2": h2,
                      "z2": self.head2(h2)})
        return state

    def forward_level3(self, state: Dict[str, torch.Tensor], x_b: torch.Tensor) -> Dict[str, torch.Tensor]:
        fb, hb = self.backbone_b(x_b)
        fused = self.fusion(state["f2"], fb)
        z3 = self.head3(torch.cat([fused, state["h2"], hb], dim=1))
        state.update({"fb": fb, "hb": hb, "fused": fused, "z3": z3})
        return state

    def route(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        probs = state["z1"].softmax(dim=1)
        ent = self.entropy(probs)
        unc = self.evidential_uncertainty(state["z1"])
        anat = state.get("anat_code")
        if anat is None:
            anat = torch.zeros(state["ha"].shape[0], self.cfg.anat_dim, device=state["ha"].device)
        if "no_anat_input" in self.cfg.ablate:
            anat = torch.zeros_like(anat)
        if "no_evid_input" in self.cfg.ablate:
            unc = torch.zeros_like(unc)
        r_in = UtilityRouter.build_input(state["ha"], probs, ent, unc, anat)
        return self.router(r_in)

    # ------------------------------------------------------------------ #
    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor, mask: torch.Tensor,
                training_mode: bool = True) -> Dict[str, torch.Tensor]:
        """Training forward: all levels evaluated, routing kept soft."""
        state = self.forward_level1(x_a)
        # the mask encoder is cheap, so its code is always available to the router
        state = self.forward_level2(state, mask)
        state = self.forward_level3(state, x_b)
        r_logits = self.route(state)
        r = torch.sigmoid(r_logits)

        # soft cascade: gate_k = P(escalate to level k) from predicted gains
        lam = self.cfg.lam
        c = self.cfg.level_costs
        g2 = torch.sigmoid(((r[:, 1] - r[:, 0]) - lam * (c[1] - c[0])) * 10.0)
        g3 = g2 * torch.sigmoid(((r[:, 2] - r[:, 1]) - lam * (c[2] - c[1])) * 10.0)
        z_soft = (
            (1 - g2).unsqueeze(1) * state["z1"]
            + (g2 - g3).unsqueeze(1) * state["z2"]
            + g3.unsqueeze(1) * state["z3"]
        )
        state.update({"router_logits": r_logits, "router_probs": r,
                      "gate2": g2, "gate3": g3, "z_soft": z_soft})
        return state

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def infer(self, x_a: torch.Tensor, x_b: torch.Tensor, mask: torch.Tensor,
              lam: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """Hard-routed inference. Levels are computed only for the samples that
        need them, so the reported FLOPs are the ones actually spent."""
        lam = self.cfg.lam if lam is None else lam
        c = self.cfg.level_costs
        B = x_a.shape[0]
        device = x_a.device

        state = self.forward_level1(x_a)
        anat_map, anat_code = self.anat_enc(mask)
        state.update({"anat_map": anat_map, "anat_code": anat_code})
        r = torch.sigmoid(self.route(state))

        level = torch.ones(B, dtype=torch.long, device=device)
        logits = state["z1"].clone()

        if "no_l2" in self.cfg.ablate:
            # binary gate: level 1 or straight to the full dual-stream level 3
            go = (r[:, 2] - r[:, 0]) > lam * (c[2] - c[0])
            if go.any():
                idx = torch.nonzero(go).squeeze(1)
                f2 = self.spatial_gate(state["fa"][idx], anat_map[idx])
                f2 = self.film(f2, anat_code[idx])
                h2 = f2.mean(dim=(2, 3))
                fb, hb = self.backbone_b(x_b[idx])
                fused = self.fusion(f2, fb)
                logits[idx] = self.head3(torch.cat([fused, h2, hb], dim=1))
                level[idx] = 3
            return {"logits": logits, "level": level, "router_probs": r,
                    "z1": state["z1"], "uncertainty": self.evidential_uncertainty(logits)}

        go2 = (r[:, 1] - r[:, 0]) > lam * (c[1] - c[0])
        if go2.any():
            sub = {"fa": state["fa"][go2], "ha": state["ha"][go2]}
            f2 = self.spatial_gate(sub["fa"], anat_map[go2])
            f2 = self.film(f2, anat_code[go2])
            h2 = f2.mean(dim=(2, 3))
            z2 = self.head2(h2)
            logits[go2] = z2
            level[go2] = 2

            go3_local = (r[go2][:, 2] - r[go2][:, 1]) > lam * (c[2] - c[1])
            if go3_local.any():
                idx = torch.nonzero(go2).squeeze(1)[go3_local]
                fb, hb = self.backbone_b(x_b[idx])
                fused = self.fusion(f2[go3_local], fb)
                z3 = self.head3(torch.cat([fused, h2[go3_local], hb], dim=1))
                logits[idx] = z3
                level[idx] = 3

        return {"logits": logits, "level": level, "router_probs": r,
                "z1": state["z1"], "uncertainty": self.evidential_uncertainty(logits)}

    # ------------------------------------------------------------------ #
    def set_budget(self, lam: float) -> None:
        """Move the operating point at deployment time (no retraining)."""
        self.cfg.lam = float(lam)
