#!/usr/bin/env python3
"""
CLIP encoder + ConvTranspose decoder pre-training
-------------------------------------------------
Launch:  python enc_dec_clip.py --cfg pretrain_cfg.yaml
"""

import argparse, re, yaml, torch, minari, numpy as np, open_clip
from types import SimpleNamespace
from pathlib import Path
from tqdm import tqdm
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ─────────────────────────────────────────────────────────────
# 1.  YAML → dot-dict helper
# ─────────────────────────────────────────────────────────────
def load_cfg(path):
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return SimpleNamespace(**{k: SimpleNamespace(**v) for k, v in raw.items()})

def slug(txt):
    return re.sub(r"[^a-zA-Z0-9]+", "_", txt.strip()).lower()

def save_name(cfg):
    fam  = slug(cfg.dataset.family)
    tail = "_".join(cfg.dataset.id.split("/")[-2:])
    return f"clip_finetune_{fam}_{slug(tail)}.pt"

# ─────────────────────────────────────────────────────────────
# 2.  Dataset
# ─────────────────────────────────────────────────────────────
class MinariFrames(Dataset):
    def __init__(self, cfg):
        self.ds     = minari.load_dataset(cfg.dataset.id)
        self.store  = self.ds.storage
        self.n_ctx  = cfg.dataset.n_ctx
        self.tfm    = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224), transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275 , 0.40821073),
                (0.26862954, 0.26130258, 0.27577711))
        ])
        self.frames = []
        for epi, ep in enumerate(self.ds.iterate_episodes()):
            if len(ep) < self.n_ctx: continue
            self.frames += [(epi, t) for t in range(len(ep) - (self.n_ctx - 1))]

    def __len__(self): return len(self.frames)

    def _rgb(self, obs):
        return np.repeat(obs[..., -1:], 3, -1) if obs.shape[-1] == 4 else obs

    def __getitem__(self, idx):
        epi, t0 = self.frames[idx]
        ep      = next(self.store.get_episodes([epi]))
        imgs = [self.tfm(self._rgb(ep["observations"][t0 + i]))
                for i in range(self.n_ctx)]
        return imgs[0], imgs[1], imgs[-1]     # anchor, pos, neg

# ─────────────────────────────────────────────────────────────
# 3.  Model blocks
# ─────────────────────────────────────────────────────────────
class Projector(nn.Module):
    def __init__(self, enc_out, proj_dim):
        super().__init__()
        self.fc = nn.Linear(enc_out, proj_dim, bias=False)
    def forward(self, x): return F.normalize(self.fc(x), dim=-1)

class ConvDecoder(nn.Module):
    def __init__(self, enc_out, img_res, h):
        super().__init__()
        self.fc = nn.Linear(enc_out, 8 * 8 * h)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(h, h//2, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(h//2, h//4, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(h//4, 3,       4, 2, 1), nn.Sigmoid())
        self.img_res = img_res
    def forward(self, z):
        x = self.fc(z).view(-1, self.deconv[0].in_channels, 8, 8)
        return self.deconv(x)

# ─────────────────────────────────────────────────────────────
# 4.  Training routine
# ─────────────────────────────────────────────────────────────
def train(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # CLIP backbone
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai")
    vision = model.visual.requires_grad_(False)
    for blk in vision.transformer.resblocks[-cfg.training.unfreeze:]:
        blk.requires_grad_(True)
    vision = vision.to(device)

    proj    = Projector(cfg.model.enc_out, cfg.model.proj_dim).to(device)
    dec     = ConvDecoder(cfg.model.enc_out,
                          cfg.model.img_res,
                          cfg.model.dec_hlat).to(device)

    data = MinariFrames(cfg)
    loader = DataLoader(data,
                        batch_size=cfg.training.batch,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=torch.cuda.is_available())

    # optimiser
    params = list(proj.parameters()) + list(dec.parameters()) + \
             [p for p in vision.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.training.lr, weight_decay=1e-4)

    for ep in range(cfg.training.epochs):
        bar = tqdm(loader, desc=f"epoch {ep+1}/{cfg.training.epochs}")
        for a, p, n in bar:
            a, p, n = a.to(device), p.to(device), n.to(device)

            with torch.no_grad():
                f_a, f_p, f_n = vision(a), vision(p), vision(n)
            z_a, z_p, z_n = proj(f_a), proj(f_p), proj(f_n)

            # InfoNCE
            pos = (z_a * z_p).sum(-1) / cfg.training.temperature
            neg = (z_a * z_n).sum(-1) / cfg.training.temperature
            loss_infonce = -torch.log(torch.sigmoid(pos - neg)).mean()

            # Reconstruction
            recon = dec(f_a.detach())
            target = F.interpolate(a, (cfg.model.img_res, cfg.model.img_res))
            loss_recon = F.mse_loss(recon, target)

            loss = loss_recon + 0.1 * loss_infonce
            opt.zero_grad(); loss.backward(); opt.step()

            bar.set_postfix(mse=f"{loss_recon:.3f}",
                            nce=f"{loss_infonce:.3f}")

    # save
    ckpt = {
        "vision": vision.state_dict(),
        "decoder": dec.state_dict(),
        "config": cfg.__dict__
    }
    ckpt_path = Path(cfg.output.base_dir) / save_name(cfg)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, ckpt_path)
    print("✓ saved to", ckpt_path)

# ─────────────────────────────────────────────────────────────
# 5.  Entry-point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="pretrain_cfg.yaml",
                   help="YAML with dataset/model/training/output sections")
    cfg = load_cfg(p.parse_args().cfg)
    train(cfg)
