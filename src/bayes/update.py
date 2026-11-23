"""Actualización bayesiana secuencial sobre el banco KDE.

Estado: log-verosimilitud acumulada l_k(c) para cada condición c.
Prior 50/50 POR CLASE (no por condición): evita el sesgo 49:10.
"""
from __future__ import annotations

import numpy as np

from src.kde.bank import KDEBank


def build_prior(bank: KDEBank,
                labels: dict[tuple, int | None],
                forced_class: dict[tuple, int] | None = None,
                prior_class: float = 0.5) -> dict[tuple, float]:
    """Prior uniforme DENTRO de cada clase.

    labels: (chsp, osnr) -> z o None.
    forced_class: mapeo opcional para asignar clase a condiciones ambiguas/control.
    """
    forced_class = forced_class or {}
    class_members: dict[int, list[tuple]] = {0: [], 1: []}
    for key in bank.keys():
        z = labels.get(key)
        if z is None:
            z = forced_class.get(key)
        if z in (0, 1):
            class_members[z].append(key)

    prior: dict[tuple, float] = {}
    for z, members in class_members.items():
        if not members:
            continue
        p = prior_class / len(members)
        for key in members:
            prior[key] = p

    for key in bank.keys():
        prior.setdefault(key, 0.0)
    total = sum(prior.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Priores no suman 1: {total}")
    return prior

def log_sum_exp(a: np.ndarray) -> float:
    """log Σ exp(a_i) estable."""
    m = np.max(a)
    if not np.isfinite(m):
        return m
    return m + np.log(np.sum(np.exp(a - m)))


def accumulate_loglik(bank: KDEBank, loglik: dict[tuple, float],
                      block: np.ndarray,
                      temperature: float = 1.0,
                      max_step: float | None = None) -> None:
    """Suma Σ_{y∈block} log f_c(y) / temperature a loglik[c] in-place.
    Permite atemperar la verosimilitud (factor de tamaño efectivo) y limitar
    la contribución por bloque con max_step."""
    for key, cond in bank.conditions.items():
        if not np.isfinite(loglik[key]):
            continue
        step = float(np.sum(cond.log_pdf(block))) / float(temperature)
        if max_step is not None:
            step = float(np.clip(step, -max_step, max_step))
        loglik[key] += step


def posterior_from_loglik(loglik: dict[tuple, float],
                          prior: dict[tuple, float]) -> dict[tuple, float]:
    """π_k(c) ∝ π_0(c) exp(ℓ_k(c)), en log-sum-exp. Devuelve dict normalizado."""
    keys = [k for k in prior if prior[k] > 0]
    logp = np.array([np.log(prior[k]) + loglik[k] for k in keys])
    z = log_sum_exp(logp)
    return {k: float(np.exp(lp - z)) for k, lp in zip(keys, logp)}


def overlap_prob(posterior: dict[tuple, float],
                 inventory_labels: dict[tuple, int]) -> float:
    """b_k = Σ_{c: z(c)=1} π_k(c)."""
    return float(sum(p for k, p in posterior.items() if inventory_labels.get(k) == 1))


def compute_brier_score(beliefs: np.ndarray, targets: np.ndarray) -> float:
    """Calcula el Brier Score: MSE entre las probabilidades predichas y las etiquetas binarias."""
    b = np.asarray(beliefs, dtype=float)
    z = np.asarray(targets, dtype=float)
    return float(np.mean((b - z) ** 2))


def reliability_curve(beliefs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula el diagrama de confiabilidad (probabilidad predicha vs frecuencia observada).

    Retorna:
        bin_preds: media de probabilidad predicha por bin
        bin_actuals: frecuencia empírica observada de z=1 por bin
        bin_counts: cantidad de muestras en cada bin
    """
    b = np.asarray(beliefs, dtype=float)
    z = np.asarray(targets, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    
    bin_preds = []
    bin_actuals = []
    bin_counts = []
    
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i+1]
        if i == n_bins - 1:
            mask = (b >= low) & (b <= high)
        else:
            mask = (b >= low) & (b < high)
        count = int(np.sum(mask))
        if count > 0:
            bin_preds.append(float(np.mean(b[mask])))
            bin_actuals.append(float(np.mean(z[mask])))
            bin_counts.append(count)
            
    return np.array(bin_preds), np.array(bin_actuals), np.array(bin_counts)
