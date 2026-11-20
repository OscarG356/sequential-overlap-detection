"""CLI: entrena el banco KDE con el split definido en configs/experiment.yaml."""
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loader import load_inventory
from src.kde.bank import build_bank, save_bank


def make_split(inv, cfg):
    """Devuelve listas de capture_id para train/val/test según cfg['split']."""
    strat = cfg["strategy"]
    if strat == "by_song":
        def song_id(capture_id):
            return capture_id.rsplit("/", 1)[-1].split("_", 1)[0]
        song = inv["capture_id"].map(song_id)
        train = inv[song.isin(cfg["train_songs"])]
        val = inv[song.isin(cfg["val_songs"])]
        test = inv[song.isin(cfg["test_songs"])]
    elif strat == "by_osnr":
        val_osnr = float(cfg["val_osnr"])
        test_osnr = [float(x) for x in cfg["test_osnr"]]
        train = inv[~inv["osnr_db"].isin([val_osnr, *test_osnr])]
        val = inv[inv["osnr_db"] == val_osnr]
        test = inv[inv["osnr_db"].isin(test_osnr)]
    elif strat == "by_chsp":
        val_chsp = [str(x) for x in cfg["val_chsp"]]
        test_chsp = [str(x) for x in cfg["test_chsp"]]
        train = inv[~inv["chsp_dir"].isin(val_chsp + test_chsp)]
        val = inv[inv["chsp_dir"].isin(val_chsp)]
        test = inv[inv["chsp_dir"].isin(test_chsp)]
    else:
        raise ValueError(f"strategy desconocida: {strat}")

    train = train[~train["is_ambiguous"] & ~train["is_control"]]
    return (train["capture_id"].tolist(),
            val["capture_id"].tolist(),
            test["capture_id"].tolist())


def main():
    with open("configs/experiment.yaml") as f:
        cfg = yaml.safe_load(f)

    inv = load_inventory("data/processed/inventory.csv")
    train_ids, val_ids, test_ids = make_split(inv, cfg["split"])
    print(f"train={len(train_ids)}  val={len(val_ids)}  test={len(test_ids)}")

    bank = build_bank(
        inv, train_ids,
        bandwidth=cfg["kde"]["bandwidth"],
        subsample=cfg["kde"]["subsample"],
        seed=cfg["kde"]["seed"],
    )
    save_bank(bank, "data/processed")
    print(f"Banco con {len(bank.keys())} condiciones guardado en data/processed/")


if __name__ == "__main__":
    main()
