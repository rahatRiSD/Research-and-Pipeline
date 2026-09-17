# ARBITER — Anatomy-Routed Budgeted Inference with Trusted Evidential Reasoning

A conditional-computation framework for medical image analysis, benchmarked
against 8 comparison models on two public datasets, with explainability for both
the *diagnosis* and the *escalation decision*.

---

## 1. What is new here

Anatomy-guided dual-stream networks run both streams on every case. SecondOpinion
gates the second stream with a binary correctness classifier. ARBITER changes
three things:

| | Prior dual-stream | SecondOpinion | **ARBITER** |
|---|---|---|---|
| Escalation granularity | none (always both) | binary | **3-rung cascade** |
| Gate target | — | P(stream A correct) | **P(level k correct) for every k → marginal utility** |
| Operating point | — | fixed at 0.5 | **λ knob, moved at deployment without retraining** |
| Uncertainty | softmax | softmax + entropy | **+ Dirichlet evidential vacuity fed back to the router** |
| Explains compute decision | no | no | **yes (router attribution)** |

**The middle rung is the core idea.** Level 2 injects anatomy as FiLM modulation
of the *same* level-1 feature map, conditioned on a 0.4M-parameter mask encoder.
It costs ~0.012 GFLOPs — about 3% of a second backbone — so cases that need only
a hint of anatomical context never pay for the full dual-stream path.

**The router predicts the right quantity.** "Is stream A correct" is not what the
decision needs; it needs the *gain* of escalating versus its price. ARBITER's
router outputs `r₁, r₂, r₃ = P(level k correct)` and escalates while

```
r_{k+1} − r_k  >  λ · (c_{k+1} − c_k)
```

Sweeping λ traces the full accuracy-vs-FLOPs Pareto front from one set of
weights, which is Figure 3 and which no fixed-threshold gate can produce.

---

## 2. Datasets

| | Dataset | Data paper | Access | Anatomy channel |
|---|---|---|---|---|
| D1 | **FracAtlas** (4,083 musculoskeletal X-rays, 717 fracture) | Abedeen et al., *Scientific Data* 2023, `10.1038/s41597-023-02432-4` | Kaggle mirror / figshare `10.6084/m9.figshare.22363012`, CC-BY 4.0 | U-Net trained on the COCO fracture masks, then inferred for all scans |
| D2 | **NIH ChestX-ray14** (single-label 6-class subset) | Wang et al., CVPR 2017 | Kaggle `nih-chest-xrays/data` | U-Net lung segmentation, inferred for all scans |

Drop-in alternative for D2: the **COVID-19 Radiography Database** ships
ground-truth lung masks for every image (`configs` → `dataset.name: covid`),
which isolates "predicted vs ground-truth anatomy" as an extra ablation.

**No annotation leakage.** Classifiers always consume *predicted* masks, on train
and test alike. Ground-truth masks are only ever seen by the segmenter.

---

## 3. Setup

```bash
pip install -r requirements.txt
```

### FracAtlas

```bash
# 1) rasterise the COCO polygons into PNG masks (segmenter supervision)
python scripts/precompute_masks.py \
    --export-fracatlas-coco /kaggle/input/fracatlas/FracAtlas \
    --out data/masks_gt/fracatlas

# 2) train the anatomy segmenter
python scripts/train_segmenter.py \
    --images /kaggle/input/fracatlas/FracAtlas/images \
    --masks  data/masks_gt/fracatlas \
    --out runs/segmenter_fracatlas --epochs 30

# 3) predict a mask for every scan
python scripts/precompute_masks.py \
    --images /kaggle/input/fracatlas/FracAtlas/images \
    --ckpt runs/segmenter_fracatlas/segmenter.pt \
    --out data/masks_pred/fracatlas
```

Skip steps 1–3 to start immediately: set `anatomy.source: otsu` in the config and
the pipeline runs end to end with a deterministic Otsu anatomy proxy. Results
will be lower — use it only to verify the plumbing.

### ChestX-ray14

Train the segmenter on any public lung-mask set (Montgomery / Shenzhen / JSRT,
all on Kaggle), then run step 3 with `--fallback lung`.

---

## 4. Running the benchmark

Smoke test first (≈10 min, exercises every code path):

```bash
python scripts/run_all.py --configs configs/fracatlas.yaml --smoke
```

Full run — 9 model configs × 2 datasets × 5 folds:

```bash
python scripts/run_all.py --configs configs/fracatlas.yaml configs/nih_cxr14.yaml
```

Single model:

```bash
python scripts/train.py --config configs/fracatlas.yaml --model arbiter
```

Ablations for Table 4:

```bash
for ab in no_l2 no_film no_spatial no_evid_input no_anat_input; do
  python scripts/train.py --config configs/fracatlas.yaml --model arbiter --ablate $ab
done
```

Outputs land in `runs/<dataset>/<model>/` (`summary.json`, per-fold
`metrics.json`, `preds.npz`, `weights.pt`, `budget_sweep.json`, `cost.json`).

---

## 5. Comparison models (8 + ours)

| | Model | Role |
|---|---|---|
| M1 | ResNet-50 | generic CNN baseline |
| M2 | DenseNet-121 | CheXNet-style baseline |
| M3 | ConvNeXt-T | modern CNN baseline |
| M4 | ViT-S/16 | transformer baseline |
| M5 | Stream A only | primary pathway ablation |
| M6 | Stream B only | anatomy pathway ablation |
| M7 | Always-On dual-stream | PelFANet-style upper bound on cost |
| M8 | SecondOpinion | prior SOTA, binary correctness gate |
| — | **ARBITER** | ours |

Every model draws its encoder from the same factory and consumes the same inputs,
so the tables isolate the method rather than the backbone.

---

## 6. Paper artefacts

`python scripts/make_tables.py` → `paper/tables/*.tex` and `*.csv`

| Table | Content |
|---|---|
| T1 | Fracture results (Acc/Prec/Rec/Spec/F1/F1_{R/S}/AUROC, 5-fold mean ± 95% CI) |
| T2 | Chest results |
| T3 | Efficiency and routing (params, GFLOPs, level mix, router agreement, oracle regret) |
| T4 | ARBITER ablations |
| T5 | Calibration (ECE, Brier) and XAI faithfulness (deletion/insertion AUC, pointing game, mask energy) |

`python scripts/make_figures.py` and `python scripts/make_xai.py` → `paper/figures/`

| Figure | Content |
|---|---|
| F1 | Architecture schematic |
| F2 | Level mix per dataset + escalation rate vs task difficulty |
| F3 | Accuracy-vs-GFLOPs Pareto front from the λ sweep, with baselines plotted as points |
| F4 | Per-case XAI panel: image, mask, level-1 CAM, deep-level CAM, cross-attention |
| F5 | Reliability diagrams |
| F6 | Confusion matrices + accuracy by routed level |

---

## 7. Explainability

Four views, in `src/arbiter/xai.py`:

1. **Grad-CAM / Grad-CAM++ / XGrad-CAM / Score-CAM** on any level's feature map —
   comparing level 1 with the escalated level on the same image shows exactly
   what anatomical conditioning bought.
2. **Cross-attention maps** — where the primary stream looked inside the anatomy
   stream when it asked for help.
3. **Router attribution** — gradient × input over the router's semantic input
   blocks (features / probabilities / entropy / evidential uncertainty / anatomy
   code). This explains the *compute* decision, which confidence-gated baselines
   cannot do.
4. **Faithfulness scoring** — deletion/insertion AUC, pointing game and mask
   energy ratio, so the qualitative panel is backed by numbers instead of
   cherry-picked cases.

---

## 8. Reporting notes

- 5-fold stratified CV; patient-grouped on ChestX-ray14 to prevent leakage.
- Mean ± 95% CI over folds; bootstrap CIs also computed per fold.
- `metrics.delong_test` gives a p-value for AUROC differences against a baseline —
  use it before claiming an improvement of a few tenths of a point.
- Reported GFLOPs cover the classification pipeline only; the segmenter runs
  offline, and this should be stated explicitly in the paper.
- The λ sweep means "our accuracy" is a curve, not a point. Report the operating
  point you chose and why.

---

## 9. Limitations to state in the paper

The oracle regret column in T3 bounds how much of the remaining gap is a routing
problem versus a representation problem. If regret is small, a better router will
not help and the honest conclusion is that the deep level itself is the ceiling.
The single-label restriction on ChestX-ray14 discards genuinely multi-label
cases, and the anatomy channel on both datasets is predicted, so segmentation
error propagates into every anatomy-guided result.
