"""Explora constelaciones y amplitudes de las adquisiciones IQ."""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loader import load_inventory, load_capture, split_into_blocks


INV = Path("data/processed/inventory.csv")
OUT = Path("data/processed")
inv = load_inventory(INV)
OUT.mkdir(parents=True, exist_ok=True)

# Una captura por separación, a OSNR cercana al valor objetivo.
target_osnr = 25.0
chsp_values = sorted(inv["chsp_ghz"].dropna().unique())
fig, axes = plt.subplots(4, 5, figsize=(16, 12), sharex=True, sharey=True)
for ax, chsp in zip(axes.ravel(), chsp_values):
    rows = inv[inv["chsp_ghz"] == chsp]
    row = rows.iloc[(rows["osnr_db"] - target_osnr).abs().argmin()]
    cap = load_capture(row)
    y = cap.y[:5000]
    ax.scatter(y.real, y.imag, s=1, alpha=0.3)
    ax.set_title(f"{chsp:g} GHz · OSNR={cap.osnr_db:g} dB", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
for ax in axes.ravel()[len(chsp_values):]:
    ax.set_visible(False)
fig.suptitle("Constelaciones IQ por separación (hasta 5k símbolos)")
fig.tight_layout()
fig.savefig(OUT / "fig_constelaciones_por_chsp.png", dpi=120, bbox_inches="tight")

# Barrido de OSNR para la separación disponible más cercana al umbral geométrico.
boundary = 32.0 * 1.1
nearest_chsp = min(chsp_values, key=lambda x: abs(float(x) - boundary))
sub = inv[inv["chsp_ghz"] == nearest_chsp].sort_values("osnr_db").drop_duplicates("osnr_db")
fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharex=True, sharey=True)
for ax, (_, row) in zip(axes.ravel(), sub.iterrows()):
    cap = load_capture(row)
    y = cap.y[:5000]
    ax.scatter(y.real, y.imag, s=1, alpha=0.3)
    ax.set_title(f"OSNR={cap.osnr_db:g} dB", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
for ax in axes.ravel()[min(len(sub), 10):]:
    ax.set_visible(False)
fig.suptitle(f"Constelaciones a ChSp={nearest_chsp:g} GHz")
fig.tight_layout()
fig.savefig(OUT / "fig_constelaciones_osnr_barrido.png", dpi=120, bbox_inches="tight")

cap = load_capture(inv.iloc[0])
blocks = split_into_blocks(cap.y, m=100, disjoint=True)
print(f"{cap.capture_id}: {len(cap.y)} símbolos -> {len(blocks)} bloques de 100")

fig, ax = plt.subplots(figsize=(10, 5))
for chsp in np.linspace(min(chsp_values), max(chsp_values), 6):
    selected = min(chsp_values, key=lambda x: abs(float(x) - chsp))
    row = inv[inv["chsp_ghz"] == selected].iloc[0]
    cap = load_capture(row)
    ax.hist(np.abs(cap.y), bins=80, alpha=0.4, density=True, label=f"{selected:g} GHz")
ax.set_title("Distribución de amplitud por separación")
ax.set_xlabel("|y| normalizado")
ax.set_ylabel("densidad")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig_modulo_hist.png", dpi=120, bbox_inches="tight")
