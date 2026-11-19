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
                      block: np.ndarray) -> None:
    """Suma Σ_{y∈block} log f_c(y) a loglik[c] in-place."""
    for key, cond in bank.conditions.items():
        if not np.isfinite(loglik[key]):
            continue
        loglik[key] += float(np.sum(cond.log_pdf(block)))


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