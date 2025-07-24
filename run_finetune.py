
import argparse, subprocess, importlib, yaml, re, minari
from pathlib import Path

def slug(text): return re.sub(r"[^a-zA-Z0-9]+", "_", text.lower())

def download_minari(game):
    ds_id = f"atari/{game}/expert-v0"
    if not minari.dataset_exists(ds_id):
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
