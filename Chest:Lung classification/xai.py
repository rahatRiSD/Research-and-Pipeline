"""Explainability.

Four complementary views, because a routed model needs explanations of *two*
different decisions — the diagnosis and the escalation:

  1. Grad-CAM / Grad-CAM++ / Score-CAM over each level's feature map. Comparing
     the level-1 map with the level-2/3 map on the *same* image shows exactly
     what anatomical conditioning bought (Figure 4).
  2. Cross-attention maps from the fusion module: where the primary stream looks
     in the anatomy stream when it asks for help.
  3. Router attribution: gradient x input over the router's input blocks
     (features / probabilities / entropy / evidential uncertainty / anatomy
     code), answering "why did this case get escalated?" — an explanation of the
     compute decision, which confidence-gated baselines cannot provide.
  4. Faithfulness scoring: deletion/insertion AUC and a mask-pointing game
     against the anatomy mask, so the qualitative panels are backed by numbers
     rather than by cherry-picked examples (Table 5).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# CAM family
# --------------------------------------------------------------------------- #
class ActivationGradientHook:
    def __init__(self, module: nn.Module):
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.h1 = module.register_forward_hook(self._fwd)
        self.h2 = module.register_full_backward_hook(self._bwd)

    def _fwd(self, _m, _i, o):
        self.activations = o.detach() if torch.is_tensor(o) else o[0].detach()

    def _bwd(self, _m, _gi, go):
        self.gradients = go[0].detach()

    def close(self):
        self.h1.remove()
        self.h2.remove()


def _normalize_cam(cam: torch.Tensor, size: Tuple[int, int]) -> np.ndarray:
    cam = F.relu(cam)
    cam = F.interpolate(cam.unsqueeze(1), size=size, mode="bilinear", align_corners=False).squeeze(1)
    cam = cam - cam.amin(dim=(1, 2), keepdim=True)
    cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
    return cam.cpu().numpy()


def grad_cam(model: nn.Module, target_layer: nn.Module, forward_fn: Callable[[], torch.Tensor],
             class_idx: torch.Tensor, size: Tuple[int, int] = (224, 224),
             variant: str = "gradcam") -> np.ndarray:
    """Generic CAM over any layer of any model in this repo.

    forward_fn must run the model and return logits (B, C); class_idx selects the
    explained class per sample.
    """
    hook = ActivationGradientHook(target_layer)
    was_training = model.training
    model.eval()
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = forward_fn()
            score = logits.gather(1, class_idx.view(-1, 1)).sum()
            score.backward()
        acts, grads = hook.activations, hook.gradients
        if acts is None or grads is None:
            raise RuntimeError("CAM hooks captured nothing — check target_layer")
        if variant == "gradcam":
            weights = grads.mean(dim=(2, 3), keepdim=True)
        elif variant == "gradcam++":
            g2, g3 = grads ** 2, grads ** 3
            denom = 2 * g2 + (acts * g3).sum(dim=(2, 3), keepdim=True)
            alpha = g2 / torch.where(denom != 0, denom, torch.ones_like(denom))
            weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)
        elif variant == "xgradcam":
            norm = acts.sum(dim=(2, 3), keepdim=True) + 1e-8
            weights = (grads * acts / norm).sum(dim=(2, 3), keepdim=True)
        else:
            raise ValueError(variant)
        cam = (weights * acts).sum(dim=1)
        return _normalize_cam(cam, size)
    finally:
        hook.close()
        if was_training:
            model.train()


@torch.no_grad()
def score_cam(model: nn.Module, target_layer: nn.Module, forward_with_input: Callable[[torch.Tensor], torch.Tensor],
              x: torch.Tensor, class_idx: torch.Tensor, top_k: int = 32) -> np.ndarray:
    """Gradient-free CAM; slower but immune to gradient saturation."""
    acts: List[torch.Tensor] = []
    h = target_layer.register_forward_hook(lambda _m, _i, o: acts.append(o.detach()))
    try:
        forward_with_input(x)
        a = acts[0]  # (B, K, h, w)
        b, k, _, _ = a.shape
        energy = a.sum(dim=(2, 3))
        idx = energy.topk(min(top_k, k), dim=1).indices
        cams = torch.zeros(b, *x.shape[-2:], device=x.device)
        for j in range(idx.shape[1]):
            chan = a[torch.arange(b), idx[:, j]].unsqueeze(1)
            up = F.interpolate(chan, size=x.shape[-2:], mode="bilinear", align_corners=False)
            up = up - up.amin(dim=(2, 3), keepdim=True)
            up = up / (up.amax(dim=(2, 3), keepdim=True) + 1e-8)
            logits = forward_with_input(x * up)
            w = logits.softmax(1).gather(1, class_idx.view(-1, 1)).view(-1, 1, 1)
            cams += w * up.squeeze(1)
        cams = cams - cams.amin(dim=(1, 2), keepdim=True)
        return (cams / (cams.amax(dim=(1, 2), keepdim=True) + 1e-8)).cpu().numpy()
    finally:
        h.remove()


# --------------------------------------------------------------------------- #
# Cross-attention map
# --------------------------------------------------------------------------- #
def cross_attention_map(fusion_module, size: Tuple[int, int] = (224, 224)) -> Optional[np.ndarray]:
    """Where in the anatomy stream the primary stream attended (averaged over
    heads and query positions)."""
    attn = getattr(fusion_module, "last_attn", None)
    if attn is None:
        return None
    a = attn.mean(dim=1).mean(dim=1)         # (B, M)
    b, m = a.shape
    side = int(round(m ** 0.5))
    a = a.view(b, 1, side, side)
    a = F.interpolate(a, size=size, mode="bilinear", align_corners=False).squeeze(1)
    a = a - a.amin(dim=(1, 2), keepdim=True)
    return (a / (a.amax(dim=(1, 2), keepdim=True) + 1e-8)).cpu().numpy()


# --------------------------------------------------------------------------- #
# Router attribution
# --------------------------------------------------------------------------- #
ROUTER_BLOCKS = ("features", "probs", "entropy", "evidential_u", "anatomy")


def router_attribution(model, x_a: torch.Tensor, x_b: torch.Tensor, mask: torch.Tensor,
                       level: int = 2) -> Dict[str, np.ndarray]:
    """Gradient x input over the router input, grouped into semantic blocks.

    Returns per-sample normalised contribution of each block to the predicted
    correctness of `level`, i.e. an explanation of the escalation decision.
    """
    model.eval()
    state = model.forward_level1(x_a)
    anat_map, anat_code = model.anat_enc(mask)
    state.update({"anat_map": anat_map, "anat_code": anat_code})

    probs = state["z1"].softmax(1)
    ent = model.entropy(probs)
    unc = model.evidential_uncertainty(state["z1"])
    r_in = torch.cat([state["ha"], probs, ent.unsqueeze(1), unc.unsqueeze(1), anat_code], dim=1)
    r_in = r_in.detach().requires_grad_(True)

    logits = model.router(r_in)
    logits[:, level - 1].sum().backward()
    contrib = (r_in.grad * r_in).detach()

    d_feat = state["ha"].shape[1]
    d_prob = probs.shape[1]
    spans = {
        "features": (0, d_feat),
        "probs": (d_feat, d_feat + d_prob),
        "entropy": (d_feat + d_prob, d_feat + d_prob + 1),
        "evidential_u": (d_feat + d_prob + 1, d_feat + d_prob + 2),
        "anatomy": (d_feat + d_prob + 2, r_in.shape[1]),
    }
    out = {k: contrib[:, a:b].sum(1).cpu().numpy() for k, (a, b) in spans.items()}
    total = np.abs(np.stack(list(out.values()), 1)).sum(1, keepdims=True) + 1e-8
    return {k: v / total.squeeze(1) for k, v in out.items()}


# --------------------------------------------------------------------------- #
# Faithfulness
# --------------------------------------------------------------------------- #
@torch.no_grad()
def deletion_insertion_auc(forward_fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor,
                           cam: np.ndarray, class_idx: torch.Tensor, steps: int = 20,
                           mode: str = "deletion", baseline: str = "blur") -> np.ndarray:
    """Per-sample AUC of the class probability as pixels are removed (deletion,
    lower is better) or added back (insertion, higher is better) in CAM order."""
    b, c, h, w = x.shape
    heat = torch.from_numpy(cam).to(x.device).view(b, -1)
    order = heat.argsort(dim=1, descending=True)
    if baseline == "blur":
        k = 21
        pad = k // 2
        blur = F.avg_pool2d(F.pad(x, (pad,) * 4, mode="reflect"), k, stride=1)
    else:
        blur = torch.zeros_like(x)

    n_pix = h * w
    per_step = max(n_pix // steps, 1)
    scores = torch.zeros(b, steps + 1, device=x.device)
    cur = x.clone() if mode == "deletion" else blur.clone()
    src = blur if mode == "deletion" else x

    logits = forward_fn(cur)
    scores[:, 0] = logits.softmax(1).gather(1, class_idx.view(-1, 1)).squeeze(1)
    flat_cur = cur.view(b, c, -1)
    flat_src = src.view(b, c, -1)
    for s in range(steps):
        sel = order[:, s * per_step:(s + 1) * per_step]
        for ch in range(c):
            flat_cur[:, ch].scatter_(1, sel, flat_src[:, ch].gather(1, sel))
        logits = forward_fn(flat_cur.view(b, c, h, w))
        scores[:, s + 1] = logits.softmax(1).gather(1, class_idx.view(-1, 1)).squeeze(1)
    return scores.mean(dim=1).cpu().numpy()


def pointing_game(cam: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """1 if the CAM peak falls inside the anatomy mask. A blunt but standard
    check that attention is anatomically plausible rather than on collimation
    borders or burned-in markers."""
    b = cam.shape[0]
    hits = np.zeros(b)
    for i in range(b):
        flat = cam[i].reshape(-1)
        pk = int(flat.argmax())
        hits[i] = float(mask[i].reshape(-1)[pk] > 0.5)
    return hits


def mask_energy_ratio(cam: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fraction of CAM energy inside the anatomy mask (a soft pointing game)."""
    num = (cam * mask).reshape(cam.shape[0], -1).sum(1)
    den = cam.reshape(cam.shape[0], -1).sum(1) + 1e-8
    return num / den


# --------------------------------------------------------------------------- #
# Target-layer resolution
# --------------------------------------------------------------------------- #
def resolve_target_layer(model: nn.Module, which: str = "a") -> nn.Module:
    """Last conv-ish module of the requested stream."""
    if hasattr(model, "backbone_a") and which == "a":
        root = model.backbone_a.model
    elif hasattr(model, "backbone_b") and which == "b":
        root = model.backbone_b.model
    elif hasattr(model, "backbone"):
        root = model.backbone.model
    else:
        root = model
    convs = [m for m in root.modules() if isinstance(m, nn.Conv2d)]
    if convs:
        return convs[-1]
    norms = [m for m in root.modules() if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm))]
    if norms:
        return norms[-1]
    raise RuntimeError("no suitable CAM target layer found")


def overlay(cam: np.ndarray, gray: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Jet-style overlay without requiring matplotlib at call time."""
    import matplotlib.cm as cm

    heat = cm.jet(cam)[..., :3]
    base = np.stack([gray] * 3, axis=-1)
    return np.clip((1 - alpha) * base + alpha * heat, 0, 1)
