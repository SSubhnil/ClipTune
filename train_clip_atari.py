#!/usr/bin/env python3
# Lightweight fine-tuning of CLIP visual encoder on offline Atari frames
# Works with torch ≥2.0, torchvision ≥0.15, transformers ≥4.40 and open_clip_torch ≥2.24

import argparse, os, random, torch, glob
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F, Module
from torch.optim import AdamW
import minari
import numpy as np

import open_clip  # pip install open_clip_torch


# ---------- 1.  Dataset ---------- #
# ---------- 1. Dataset ---------- #
class AtariFrames(Dataset):
    """
    Streams RGB frames from a Minari offline-RL dataset and forms
    (anchor, positive, negative) triples.
    """

    def __init__(self, minari_id: str, n_ctx: int = 3):
        assert n_ctx >= 3, "Need at least 3 stacked frames"
        self.ds = minari.load_dataset(minari_id)
        self.store = self.ds.storage  # <-- NEW  (MinariStorage handle)
        self.n_ctx = n_ctx

        self.tfm = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)),
        ])

        # flat index: (episode_idx, frame_t)
        self.frames = []
        for epi, ep in enumerate(self.ds.iterate_episodes()):
            if len(ep) < self.n_ctx:
                continue
            self.frames.extend([(epi, t)
                                for t in range(len(ep) - (self.n_ctx - 1))])

    def __len__(self):
        return len(self.frames)

    def _rgb(self, obs: np.ndarray) -> np.ndarray:
        if obs.shape[-1] == 4:  # stacked gray
            g = obs[..., -1]
            return np.repeat(g[..., None], 3, axis=-1)
        if obs.shape[-1] == 3:  # native RGB
            return obs
        raise ValueError(f"Bad obs shape {obs.shape}")

    def __getitem__(self, idx):
        epi, t0 = self.frames[idx]

        # fetch a single EpisodeData via MinariStorage (returns list)
        ep = next(self.store.get_episodes([epi]))  # <-- FIX[2]

        imgs = [self.tfm(self._rgb(ep["observations"][t0 + i]))
                for i in range(self.n_ctx)]

        anchor, pos, neg = imgs[0], imgs[1], imgs[-1]
        return anchor, pos, neg


# ---------- 2.  Small projection head ---------- #
class Projector(Module):
    def __init__(self, in_dim=512, out_dim=128):
        super().__init__()
        self.fc = torch.nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):  # x: (B, in_dim)
        return F.normalize(self.fc(x), dim=-1)


# ---------- 3.  Training ---------- #
def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load CLIP ViT-B/32
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    vision = model.visual  # visual encoder only
    vision.requires_grad_(False)  # freeze
    # Unfreeze last x transformer blocks for light fine-tune
    if args.unfreeze > 0:
        for blk in vision.transformer.resblocks[-args.unfreeze:]:
            blk.requires_grad_(True)

    vision.to(device)
    projector = Projector(vision.output_dim, args.proj_dim).to(device)

    dataset = AtariFrames(args.minari_id, n_ctx=3)
    loader = DataLoader(dataset,
                        batch_size=args.batch,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=torch.cuda.is_available())

    params = [p for p in projector.parameters()] + \
             [p for p in vision.parameters() if p.requires_grad]
    optim = AdamW(params, lr=args.lr, weight_decay=1e-4)

    for epoch in range(args.epochs):
        for anchor, pos, neg in tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}"):
            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)

            with torch.no_grad():
                f_a = vision(anchor)  # (B, 512)
                f_p = vision(pos)
                f_n = vision(neg)

            z_a, z_p, z_n = projector(f_a), projector(f_p), projector(f_n)

            # InfoNCE loss (temporal contrast)
            pos_sim = (z_a * z_p).sum(-1) / args.temp
            neg_sim = (z_a * z_n).sum(-1) / args.temp
            loss = -torch.log(torch.sigmoid(pos_sim - neg_sim)).mean()

            optim.zero_grad()
            loss.backward()
            optim.step()

        print(f"epoch {epoch + 1}: loss {loss.item():.4f}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({
        "vision_state_dict": vision.state_dict(),
        "projector_state_dict": projector.state_dict()
    }, out / "clip_atari_finetuned.pt")
    print(f"✔ Saved to {out / 'clip_atari_finetuned.pt'}")


# ---------- 4.  CLI ---------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--minari_id", type=str, required=True,
                   help="e.g. atari/ale-krull-v5/cleanrl-expert-v0")
    p.add_argument("--out_dir", type=str, default="checkpoints")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--proj_dim", type=int, default=128,
                   help="dimension of learned projector head")
    p.add_argument("--unfreeze", type=int, default=1,
                   help="how many last transformer blocks to fine-tune")
    p.add_argument("--temp", type=float, default=0.07,
                   help="temperature in InfoNCE")
    run(p.parse_args())

"""
python train_clip_atari.py \
       --minari_id atari/krull/expert-v0 \
       --epochs 20 \
       --batch 64 \
       --unfreeze 2
"""