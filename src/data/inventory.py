"""Inventario de adquisiciones: recorre el árbol, parsea (ChSp, OSNR)
y produce un DataFrame reproducible con la etiqueta geométrica."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

# --- parseo de nombres de archivo ---
# Ej: "16QAM_3x16GBd_consY23dB.csv" -> osnr=23.0
#     "16QAM_3x16GBd_consY21.5dB.csv" -> osnr=21.5
_OSNR_RE = re.compile(r"consY(?P<osnr>[\d.]+)dB")

def _parse_osnr(fname: str) -> float | None:
    m = _OSNR_RE.search(fname)
    return float(m.group("osnr")) if m else None

def _parse_chsp_from_dir(dirname: str) -> float | str:
    """'16GHz'->16.0, '15p5GHz'->15.5, 'single_ch'->'single'."""
    if dirname == "single_ch":
        return "single"
    m = re.match(r"(?P<val>[\d p]+)GHz$", dirname)
    if not m:
        return dirname  # directorio raro, se conserva como string
    return float(m.group("val").replace("p", "."))

def _label_geometrica(chsp, boundary: float) -> int | None:
    """z=1 traslape (ChSp < B_nom), z=0 sin traslape (ChSp > B_nom).
    None si es ambiguo o control sin canal adyacente."""
    if not isinstance(chsp, float):
        return None
    if chsp == boundary:
        return None  # frontera exacta: no se etiqueta duro
    return 1 if chsp < boundary else 0

@dataclass
class InventoryConfig:
    root: Path
    boundary_ghz: float
    ambiguous_dirs: list[str]
    control_dirs: list[str]
    ignore_files: list[str]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "InventoryConfig":
        with open(path) as f:
            cfg = yaml.safe_load(f)["data"]
        return cls(
            root=Path(cfg["root"]),
            boundary_ghz=float(cfg["boundary_ghz"]),
            ambiguous_dirs=list(cfg.get("ambiguous_dirs", [])),
            control_dirs=list(cfg.get("control_dirs", [])),
            ignore_files=list(cfg.get("ignore_files", [])),
        )

def build_inventory(cfg: InventoryConfig) -> pd.DataFrame:
    rows = []
    root = cfg.root
    if not root.exists():
        raise FileNotFoundError(f"No existe root: {root}")

    for csv_path in sorted(root.rglob("*.csv")):
        rel = csv_path.relative_to(root)
        parts = rel.parts
        if len(parts) == 1:
            # archivo suelto (p. ej. 2x16QAM_16GBd.csv) -> se ignora o marca
            if csv_path.name in cfg.ignore_files:
                continue
            chsp_dir = "__root__"
        else:
            chsp_dir = parts[0]

        if csv_path.name in cfg.ignore_files:
            continue

        chsp = _parse_chsp_from_dir(chsp_dir)
        osnr = _parse_osnr(csv_path.name)

        is_ambiguous = chsp_dir in cfg.ambiguous_dirs
        is_control = chsp_dir in cfg.control_dirs or chsp == "single"

        label = None
        if not is_ambiguous and not is_control:
            label = _label_geometrica(chsp, cfg.boundary_ghz)

        rows.append({
            "capture_id": f"{chsp_dir}/{csv_path.stem}",
            "chsp_ghz": chsp,
            "chsp_dir": chsp_dir,
            "osnr_db": osnr,
            "path": str(csv_path),
            "is_ambiguous": is_ambiguous,
            "is_control": is_control,
            "label_geometrica": label,  # 1=traslape, 0=sin, None=ambiguo/control
        })

    df = pd.DataFrame(rows).sort_values(["chsp_dir", "osnr_db"]).reset_index(drop=True)
    return df

def summarize(df: pd.DataFrame) -> str:
    lines = []
    lines.append(f"Total adquisiciones: {len(df)}")
    lines.append(f"  con etiqueta dura (z=0/1): {df['label_geometrica'].notna().sum()}")
    lines.append(f"  ambiguas (frontera):       {df['is_ambiguous'].sum()}")
    lines.append(f"  control sin interferencia: {df['is_control'].sum()}")
    lines.append("")
    lines.append("Por directorio (ChSp):")
    for d, g in df.groupby("chsp_dir", sort=False):
        zs = g["label_geometrica"].dropna().unique().tolist()
        lines.append(f"  {d:12s} n={len(g):2d}  OSNR=[{g['osnr_db'].min()}..{g['osnr_db'].max()}]  z={zs}")
    return "\n".join(lines)