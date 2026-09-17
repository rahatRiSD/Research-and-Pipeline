#!/usr/bin/env python3
"""Run the whole benchmark: 8 models x 2 datasets, then tables, figures and XAI.

    python scripts/run_all.py --configs configs/fracatlas.yaml configs/nih_cxr14.yaml

Use --smoke for a 2-epoch, 400-image dry run that exercises every code path in
a few minutes before committing GPU hours.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODELS = [
    "resnet50",             # M1 generic CNN baseline
    "densenet121",          # M2 CheXNet-style baseline
    "convnext_tiny",        # M3 modern CNN baseline
    "vit_small_patch16_224",  # M4 transformer baseline
    "stream_a",             # M5 primary stream only (ablation)
    "stream_b",             # M6 anatomy stream only (ablation)
    "always_on",            # M7 unconditional dual-stream (PelFANet-style)
    "second_opinion",       # M8 prior SOTA: binary correctness gate
    "arbiter",              # ours
]


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    t0 = time.time()
    r = subprocess.run(cmd)
    print(f"  -> exit {r.returncode} in {time.time() - t0:.0f}s", flush=True)
    if r.returncode != 0:
        print("  !! failed; continuing with the remaining runs", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-xai", action="store_true")
    args = ap.parse_args()

    py = sys.executable
    for cfg in args.configs:
        for m in args.models:
            cmd = [py, str(ROOT / "scripts" / "train.py"), "--config", cfg,
                   "--model", m, "--folds", str(2 if args.smoke else args.folds),
                   "--out", args.out]
            if args.smoke:
                cmd += ["--epochs", "2", "--limit", "400"]
            run(cmd)

    run([py, str(ROOT / "scripts" / "make_tables.py"), "--runs", args.out, "--out", "paper/tables"])
    run([py, str(ROOT / "scripts" / "make_figures.py"), "--runs", args.out, "--out", "paper/figures"])
    if not args.skip_xai:
        for cfg in args.configs:
            run([py, str(ROOT / "scripts" / "make_xai.py"), "--config", cfg,
                 "--runs", args.out, "--out", "paper/figures"])
    print("\nAll done. Tables in paper/tables, figures in paper/figures.")


if __name__ == "__main__":
    main()
