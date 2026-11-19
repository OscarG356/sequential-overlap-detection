"""Exploración: constelaciones por ChSp, histogramas I/Q, efecto de OSNR."""
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loader import load_inventory, load_capture, split_into_blocks

INV = "data/processed/inventory.csv"
inv = load_inventory(INV)

# --- Figura 1: constelaciones por ChSp a OSNR fija (23 dB) ---
osnr_fija = 23.0
sub = inv[inv["osnr_db"] == osnr_fija].sort_values("chsp_ghz", key=lambda s: s.astype(str))
fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
axes = axes.ravel()
for ax, (_, row) in zip(axes, sub.iterrows()):
    cap = load_capture(row)
    y = cap.y[:5000]  # muestra para graficar
    ax.scatter(y.real, y.imag, s=1, alpha=0.3)
    ax.set_title(f"{cap.chsp_ghz} GHz\nOSNR={cap.osnr_db} dB", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
fig.suptitle(f"Constelaciones 16-QAM a OSNR={osnr_fija} dB (5k símbolos)", y=1.02)
fig.tight_layout()
fig.savefig("data/processed/fig_constelaciones_osnr23.png", dpi=120, bbox_inches="tight")
print("Guardada fig_constelaciones_osnr23.png")

# --- Figura 2: efecto de OSNR a ChSp=16 GHz ---
sub = inv[inv["chsp_dir"] == "16GHz"].sort_values("osnr_db")
fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharex=True, sharey=True)
axes = axes.ravel()
for ax, (_, row) in zip(axes, sub.iterrows()):
    cap = load_capture(row)
    y = cap.y[:5000]
    ax.scatter(y.real, y.imag, s=1, alpha=0.3)
    ax.set_title(f"OSNR={cap.osnr_db} dB", fontsize=9)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
fig.suptitle("Constelaciones a ChSp=16 GHz vs OSNR", y=1.02)
fig.tight_layout()
fig.savefig("data/processed/fig_constelaciones_osnr_barrido.png", dpi=120, bbox_inches="tight")
print("Guardada fig_constelaciones_osnr_barrido.png")

# --- Figura 3: bloques disjuntos de una captura ---
cap = load_capture(inv[inv["capture_id"] == "17GHz/16QAM_3x16GBd_consY23dB"].iloc[0])
blocks = split_into_blocks(cap.y, m=100, disjoint=True)
print(f"{cap.capture_id}: {len(cap.y)} símbolos -> {len(blocks)} bloques de 100")

# --- Diagnóstico: distribución de |y| por ChSp ---
fig, ax = plt.subplots(figsize=(10, 5))
for chsp in ["15GHz", "16GHz", "17GHz", "17p6GHz", "18GHz", "single_ch"]:
    rows = inv[inv["chsp_dir"] == chsp]
    if len(rows) == 0: continue
    row = rows.iloc[0]
    cap = load_capture(row)
    ax.hist(np.abs(cap.y), bins=80, alpha=0.4, density=True, label=f"{chsp}")
ax.set_title("Distribución de |y| normalizada por condición (1er OSNR disponible)")
ax.set_xlabel("|y|"); ax.set_ylabel("densidad"); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("data/processed/fig_modulo_hist.png", dpi=120, bbox_inches="tight")
print("Guardada fig_modulo_hist.png")