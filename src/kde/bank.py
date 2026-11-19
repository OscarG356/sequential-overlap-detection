"""Banco de KDE gaussianos 2D, uno por condición c = (chsp, osnr).

Se entrena SOLO con adquisiciones de train (ver split en configs).
Devuelve log-densidad por símbolo para la actualización bayesiana.

Inferencia con numpy puro (log_kde_fast) porque sklearn KernelDensity
es ~50-100x más lento en 2D para consultas masivas.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.loader import Capture, load_capture


def log_kde_fast(y: np.ndarray, components: np.ndarray,
                 bandwidth: float) -> np.ndarray:
    """log KDE gaussiano 2D, vectorizado con numpy.

    y: complejo (N,)
    components: complejo (M,)  -- centros del KDE
    bandwidth: h (ancho de banda absoluto, isotrópico)

    Devuelve log f(y) para cada y_i.
    """
    N, M = len(y), len(components)
    out = np.empty(N, dtype=np.float64)
    log_norm = -np.log(M) - np.log(2.0 * np.pi * bandwidth ** 2)
    inv2h2 = 1.0 / (2.0 * bandwidth ** 2)
    chunk = 4096
    for i in range(0, N, chunk):
        yi = y[i:i + chunk]                                   # (B,)
        d2 = np.abs(yi[:, None] - components[None, :]) ** 2   # (B, M)
        m = d2.min(axis=1)
        out[i:i + chunk] = log_norm + (
            -m * inv2h2
            + np.log(np.exp(-(d2 - m[:, None]) * inv2h2).sum(axis=1))
        )
    return out


@dataclass
class ConditionKDE:
    chsp_ghz: float | str
    osnr_db: float
    components: np.ndarray        # (M,) complejo: centros del KDE
    n_train_symbols: int
    bandwidth: float              # h absoluto

    def log_pdf(self, y: np.ndarray) -> np.ndarray:
        return log_kde_fast(y, self.components, self.bandwidth)


@dataclass
class KDEBank:
    conditions: dict[tuple, ConditionKDE] = field(default_factory=dict)

    def add(self, cond: ConditionKDE) -> None:
        self.conditions[(cond.chsp_ghz, cond.osnr_db)] = cond

    def get(self, chsp_ghz, osnr_db) -> ConditionKDE:
        return self.conditions[(chsp_ghz, osnr_db)]

    def keys(self) -> list[tuple]:
        return list(self.conditions.keys())

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "KDEBank":
        return joblib.load(path)


def _fit_kde_on_capture(cap: Capture,
                        bandwidth: float = 0.15,
                        subsample: int | None = 3000,
                        seed: int = 0) -> tuple[np.ndarray, float]:
    """Devuelve la submuestra compleja (centros) y el bandwidth."""
    y = cap.y
    if subsample is not None and len(y) > subsample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y), size=subsample, replace=False)
        y = y[idx]
    return y.copy(), float(bandwidth)


def build_bank(inventory: pd.DataFrame,
               train_capture_ids: list[str],
               bandwidth: float = 0.15,
               subsample: int | None = 3000,
               seed: int = 0) -> KDEBank:
    """Entrena un ConditionKDE por (chsp, osnr) usando solo train_capture_ids."""
    inv = inventory[inventory["capture_id"].isin(train_capture_ids)].copy()
    if len(inv) == 0:
        raise ValueError("train_capture_ids no matchea ninguna fila del inventario")

    bank = KDEBank()
    for (chsp, osnr), group in tqdm(inv.groupby(["chsp_ghz", "osnr_db"]),
                                     desc="KDE por condición"):
        if len(group) != 1:
            raise ValueError(f"Condición ({chsp},{osnr}) tiene {len(group)} capturas; "
                             f"el banco asume 1.")
        cap = load_capture(group.iloc[0])
        components, bw = _fit_kde_on_capture(cap, bandwidth=bandwidth,
                                             subsample=subsample, seed=seed)
        bank.add(ConditionKDE(
            chsp_ghz=cap.chsp_ghz,
            osnr_db=cap.osnr_db,
            components=components,
            n_train_symbols=len(cap.y),
            bandwidth=bw,
        ))
    return bank


def save_bank(bank: KDEBank, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bank.save(out / "kde_bank.pkl")
    meta = [
        {"chsp_ghz": c.chsp_ghz, "osnr_db": c.osnr_db,
         "n_train_symbols": c.n_train_symbols, "bandwidth": c.bandwidth,
         "n_components": int(len(c.components))}
        for c in bank.conditions.values()
    ]
    (out / "kde_bank_meta.json").write_text(json.dumps(meta, indent=2))