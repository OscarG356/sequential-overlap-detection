"""CLI: construye el inventario desde el árbol de datos y lo guarda en CSV."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.inventory import InventoryConfig, build_inventory, summarize

def main():
    cfg = InventoryConfig.from_yaml("configs/experiment.yaml")
    df = build_inventory(cfg)
    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "inventory.csv", index=False)
    print(summarize(df))
    print(f"\nGuardado en {out / 'inventory.csv'}")

if __name__ == "__main__":
    main()