"""CLI: calcula la traza b_k(n) para capturas de test y grafica.

Uso:
    python scripts/trace_bk.py --capture-id "16GHz/16QAM_3x16GBd_consY35dB"
    python scripts/trace_bk.py --all-test
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.bayes.update import (build_prior, accumulate_loglik,
                              posterior_from_loglik, overlap_prob)
from src.data.loader import load_capture, split_into_blocks, load_inventory
from src.kde.bank import KDEBank
from scripts.build_kde_bank import make_split


def labels_from_inventory(inv: pd.DataFrame) -> dict[tuple, int | None]:
    out = {}
    for _, row in inv.drop_duplicates(["chsp_ghz", "osnr_db"]).iterrows():
        z = None if pd.isna(row["label_geometrica"]) else int(row["label_geometrica"])
        out[(row["chsp_ghz"], row["osnr_db"])] = z
    return out


def trace_bk(capture_row: pd.Series, bank: KDEBank,
             prior: dict[tuple, float], labels: dict[tuple, int],
             block_size: int) -> tuple[np.ndarray, np.ndarray]:
    cap = load_capture(capture_row)
    y = cap.y
    keys = list(bank.keys())
    n_cond = len(keys)

    # 1) log f_c(y_i) para todos los símbolos y todas las condiciones
    L = np.empty((len(y), n_cond), dtype=np.float64)
    for j, cond in enumerate(bank.conditions.values()):
        L[:, j] = cond.log_pdf(y)            # 1 llamada por condición

    # 2) acumular por bloques con cumsum
    cumL = np.cumsum(L, axis=0)
    n_blocks = len(y) // block_size
    ns = np.arange(1, n_blocks + 1) * block_size
    bs = np.empty(n_blocks)
    for k, n in enumerate(ns):
        loglik_k = dict(zip(keys, cumL[n - 1]))
        post = posterior_from_loglik(loglik_k, prior)
        bs[k] = overlap_prob(post, labels)
    return ns, bs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-id", type=str, default=None)
    ap.add_argument("--all-test", action="store_true")
    ap.add_argument("--block-size", type=int, default=100)
    args = ap.parse_args()

    with open("configs/experiment.yaml") as f:
        cfg = yaml.safe_load(f)

    inv = load_inventory("data/processed/inventory.csv")
    labels = labels_from_inventory(inv)          # (chsp, osnr) -> z | None
    bank = KDEBank.load("data/processed/kde_bank.pkl")

    # Prior SOLO sobre condiciones de train (las que están en el banco).
    # Las condiciones de test NO están en el banco y NO deben estarlo.
    prior = build_prior(bank, labels)

    if args.all_test:
        _, _, test_ids = make_split(inv, cfg["split"])
        skipped = [cid for cid in test_ids
                   if labels.get((inv.loc[inv["capture_id"] == cid, "chsp_ghz"].iloc[0],
                                  float(inv.loc[inv["capture_id"] == cid, "osnr_db"].iloc[0]))) is None]
        test_ids = [cid for cid in test_ids if cid not in skipped]
        if skipped:
            print(f"[info] omitidas (sin etiqueta z): {skipped}")
    else:
        test_ids = [args.capture_id]

    fig, ax = plt.subplots(figsize=(9, 5))
    for cid in test_ids:
        row = inv.loc[inv["capture_id"] == cid].iloc[0]
        ns, bs = trace_bk(row, bank, prior, labels, args.block_size)
        z = labels.get((row["chsp_ghz"], float(row["osnr_db"])))
        color = {1: "tab:orange", 0: "tab:green"}[z]
        ax.plot(ns, bs, color=color, alpha=0.7,
                label=f"{cid.split('/')[-1]} (z={z})")
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set_xlabel("Símbolos observados acumulados")
    ax.set_ylabel("$b_k = P(\\mathrm{traslape}\\,|\\,\\mathrm{datos})$")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out = "data/processed/fig_bk_trace.png"
    fig.savefig(out, dpi=120)
    print(f"Guardada {out}  ({len(test_ids)} capturas trazadas)")

if __name__ == "__main__":
    main()
