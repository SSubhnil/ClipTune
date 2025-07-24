#!/usr/bin/env python3
"""
CLIP encoder + ConvTranspose decoder pre-training

Usage example
-------------
python enc_dec_clip.py \
    --cfg          pretrain_cfg.yaml \
    --dataset_id   atari/krull/expert-v0 \
    --save_prefix  krull
"""
import argparse, yaml, re, numpy as np, torch, minari, open_clip
from types import SimpleNamespace
from pathlib import Path
from tqdm import tqdm
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ---------------------------------------------------------------------
# 1. helpers
# ---------------------------------------------------------------------
def load_cfg(path: str) -> SimpleNamespace:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return SimpleNamespace(**{k: SimpleNamespace(**v) for k, v in raw.items()})

def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")

# ---------------------------------------------------------------------
# 2. dataset
# ---------------------------------------------------------------------
class MinariFrames(Dataset):
    """Streams anchor/positive/negative triples from a Minari Atari dataset."""
    def __init__(self, dataset_id: str, n_ctx: int):
        self.ds    = minari.load_dataset(dataset_id)
        self.store = self.ds.storage
        self.n_ctx = n_ctx
        self.tfm   = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224), transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(  # CLIP mean / std
                mean=(0.48145466, 0.4578275 , 0.40821073),
                std =(0.26862954, 0.26130258, 0.27577711)),
        ])

        self.frames = []
        for epi, ep in enumerate(self.ds.iterate_episodes()):
            if len(ep) < n_ctx:
                continue
            self.frames += [(epi, t) for t in range(len(ep) - (n_ctx - 1))]

    def __len__(self): return len(self.frames)

    @staticmethod
    def _rgb(obs: np.ndarray) -> np.ndarray:
        """Stacked grayscale (H×W×4) → RGB; else passthrough."""
        if obs.shape[-1] == 4:
            g = obs[..., -1]
            return np.repeat(g[..., None], 3, -1)
        return obs

    def __getitem__(self, idx):
        epi, t0 = self.frames[idx]
        ep      = next(self.store.get_episodes([epi]))
        imgs = [self.tfm(self._rgb(ep["observations"][t0 + i]))
                for i in range(self.n_ctx)]
        return imgs[0], imgs[1], imgs[-1]  # anchor, positive, negative

# ---------------------------------------------------------------------
# 3. model blocks
# ---------------------------------------------------------------------
class Projector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
    def forward(self, x):            # (B, in_dim)
        return F.normalize(self.fc(x), dim=-1)

class ConvDecoder(nn.Module):
    """1,024-D latent → 3×64×64 RGB frame."""
    def __init__(self, in_dim: int, img_res: int, h: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 8 * 8 * h)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(h,   h // 2, 4, 2, 1), nn.ReLU(),  # 8 → 16
            nn.ConvTranspose2d(h // 2, h // 4, 4, 2, 1), nn.ReLU(),  # 16 → 32
            nn.ConvTranspose2d(h // 4, 3,       4, 2, 1), nn.Sigmoid()  # 32 → 64
        )
        self.img_res = img_res
    def forward(self, z):            # (B, in_dim)
        x = self.fc(z).view(-1, self.deconv[0].in_channels, 8, 8)
        return self.deconv(x)

# ---------------------------------------------------------------------
# 4. training routine
# ---------------------------------------------------------------------
def main(cfg, dataset_id: str, save_prefix: str | None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # CLIP ViT-B/32 backbone
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    vision: nn.Module = model.visual.requires_grad_(False)
    for blk in vision.transformer.resblocks[-cfg.training.unfreeze:]:
        blk.requires_grad_(True)
    vision = vision.to(device)

    proj = Projector(cfg.model.enc_out, cfg.model.proj_dim).to(device)
    dec  = ConvDecoder(cfg.model.enc_out, cfg.model.img_res, cfg.model.dec_hlat).to(device)

    data   = MinariFrames(dataset_id, cfg.dataset.n_ctx)
    loader = DataLoader(data,
                        batch_size   = cfg.training.batch,
                        shuffle      = True,
                        num_workers  = 4,
                        pin_memory   = torch.cuda.is_available())

    params = (list(proj.parameters()) + list(dec.parameters()) +
              [p for p in vision.parameters() if p.requires_grad])
    opt = torch.optim.AdamW(params, lr=cfg.training.lr, weight_decay=1e-4)

    # --- helper: CLIP → 1,024-D latent --------------------------------
    def clip_encode(imgs: torch.Tensor) -> torch.Tensor:
        # CLS token (512 D)
        cls_vec = vision(imgs)  # (B,512)
        # Patch tokens pooled + trim to 512 D for simplicity
        patch_tokens = vision.patch_embed(imgs)            # (B,N,768)
        patch_mean   = patch_tokens.mean(1)[:, :512]       # (B,512)
        return torch.cat([cls_vec, patch_mean], dim=-1)    # (B,1024)

    for ep in range(cfg.training.epochs):
        bar = tqdm(loader, desc=f"epoch {ep+1}/{cfg.training.epochs}")
        for a, p, n in bar:
            a, p, n = a.to(device), p.to(device), n.to(device)

            with torch.no_grad():
                f_a, f_p, f_n = clip_encode(a), clip_encode(p), clip_encode(n)

            z_a, z_p, z_n = proj(f_a), proj(f_p), proj(f_n)

            # InfoNCE
            pos = (z_a * z_p).sum(-1) / cfg.training.temperature
            neg = (z_a * z_n).sum(-1) / cfg.training.temperature
            loss_nce = -torch.log(torch.sigmoid(pos - neg)).mean()

            # Reconstruction
            recon   = dec(f_a.detach())
            target  = F.interpolate(a, (cfg.model.img_res, cfg.model.img_res))
            loss_re = F.mse_loss(recon, target)

            loss = loss_re + 0.1 * loss_nce

            opt.zero_grad()
            loss.backward()
            opt.step()
            bar.set_postfix(mse=f"{loss_re:.4f}", nce=f"{loss_nce:.4f}")

    # ---------------- save ----------------
    out_dir = Path(cfg.output.base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix  = save_prefix or slug(dataset_id.split("/")[-2])
    torch.save(vision.state_dict(),  out_dir / f"encoder_{prefix}.pt")
    torch.save(dec.state_dict(),     out_dir / f"decoder_{prefix}.pt")
    print(f"✓ saved encoder_{prefix}.pt & decoder_{prefix}.pt → {out_dir}")

# ---------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg",         default="pretrain_cfg.yaml",
                        help="YAML with dataset/model/training/output sections")
    parser.add_argument("--dataset_id",  required=True,
                        help="Minari dataset ID, e.g. atari/krull/expert-v0")
    parser.add_argument("--save_prefix", default=None,
                        help="Filename prefix for saved checkpoints")
    args = parser.parse_args()

    C = load_cfg(args.cfg)
    # store dataset info inside config (used by DataSet)
    C.dataset = SimpleNamespace(family=args.dataset_id.split("/")[0],
                                id=args.dataset_id,
                                n_ctx=C.dataset.n_ctx)
    main(C, dataset_id=args.dataset_id, save_prefix=args.save_prefix)
