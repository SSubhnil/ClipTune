import argparse, subprocess, importlib, yaml, re, minari
from pathlib import Path
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def slug(text): return re.sub(r"[^a-zA-Z0-9]+", "_", text.lower())


def dataset_present(ds_id: str) -> bool:
    """True if ds_id is already downloaded locally."""
    try:
        minari.load_dataset(ds_id)
        return True
    except FileNotFoundError:
        return False


def download_minari(game: str) -> str:
    ds_id = f"atari/{game}/expert-v0"
    if not dataset_present(ds_id):  # <- changed line
        subprocess.check_call(["minari", "download", ds_id])
    return ds_id


def run_single(game, cfg_file):
    ds_id = download_minari(game)
    out_prefix = f"{slug(game)}"
    subprocess.check_call([
        "python", "enc_dec_clip.py",
        "--cfg", cfg_file,
        "--dataset_id", ds_id,
        "--save_prefix", out_prefix
    ])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--games", nargs="+", required=True,
                   help="List of Atari game names, e.g. krull breakout alien")
    p.add_argument("--cfg", default="pretrain_cfg.yaml")
    args = p.parse_args()

    for g in args.games:
        print(f"\n=== Fine-tuning for {g} ===")
        run_single(g, args.cfg)

"""
python run_finetune.py \
       --games krull breakout alien \
       --cfg   pretrain_cfg.yaml
"""