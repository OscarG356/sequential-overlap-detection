"""Carga de adquisiciones IQ y partición en bloques disjuntos.

Cada CSV/Parquet = 1 adquisición con símbolos IQ del canal central.
Normalización: potencia media unitaria (E|y|^2 = 1) por captura.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Capture:
    capture_id: str
    y: np.ndarray            # complejo, potencia media 1
    chsp_ghz: float | str
    osnr_db: float
    label: int | None        # 1 traslape, 0 sin, None ambiguo/control
    path: Path


def load_capture(row: pd.Series) -> Capture:
    """Carga una adquisición IQ (Parquet nuevo o CSV legado)."""
    path = Path(row["path"])
    df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    pairs = [("I", "Q"), ("rx_i", "rx_q")]
    pair = next((p for p in pairs if set(p).issubset(df.columns)), None)
    if pair is None:
        raise ValueError(f"{path}: no encuentro columnas IQ (I,Q o rx_i,rx_q); hay {df.columns.tolist()}")

    y = df[pair[0]].to_numpy(dtype=np.float64) + 1j * df[pair[1]].to_numpy(dtype=np.float64)

    # normalización por potencia media
    power = np.mean(np.abs(y) ** 2)
    if power <= 0:
        raise ValueError(f"{path}: potencia nula")
    y = y / np.sqrt(power)

    return Capture(
        capture_id=row["capture_id"],
        y=y,
        chsp_ghz=row["chsp_ghz"],
        osnr_db=float(row["osnr_db"]),
        label=None if pd.isna(row["label_geometrica"]) else int(row["label_geometrica"]),
        path=path,
    )


def split_into_blocks(y: np.ndarray, m: int, disjoint: bool = True) -> list[np.ndarray]:
    """Parte y en bloques de m símbolos.

    disjoint=True  -> bloques consecutivos sin solape (descarta el sobrante)
    disjoint=False -> ventana deslizante de paso 1 (útil para curvas finas)
    """
    n = len(y)
    if disjoint:
        n_blocks = n // m
        return [y[i * m:(i + 1) * m] for i in range(n_blocks)]
    # deslizante (puede ser costoso; usar solo para m pequeño)
    return [y[i:i + m] for i in range(n - m + 1)]


def load_inventory(inventory_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(inventory_csv)
    df["label_geometrica"] = df["label_geometrica"].astype("Int64")

    def _norm_chsp(v):
        if isinstance(v, str) and v != "single":
            return float(v)
        return v
    df["chsp_ghz"] = df["chsp_ghz"].map(_norm_chsp)
    df["osnr_db"] = df["osnr_db"].astype(float)
    return df
