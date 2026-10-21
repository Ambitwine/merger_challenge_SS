"""NNCLR arm (Dwibedi et al. 2021). Same ResNet backbone and augmentations as
the SimCLR arm; only the loss differs. Each view is pulled towards the nearest
neighbour of its sibling view, taken from a queue of past embeddings.

The backbone is a plain (non-steerable) ResNet, so rotation invariance has to
come from full 360-degree rotation augmentation; the small-rotation setting is
only appropriate for an equivariant backbone.

The stack is read once into RAM when it fits C.RAM_BUDGET_GB (otherwise a
node-local $TMPDIR memmap), as in the SimCLR arm.

Outputs go to models/nnclr_tng/, separate from SimCLR, so both can run at once.

    python -u nnclr_arm.py          # train, then extract + PCA
"""
from __future__ import annotations
import json
import logging
import math
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset

import config_nnclr as C

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Full-state resume checkpoint (model + opt + sched + scaler + bookkeeping),
# written every epoch so a TIMEOUT-killed run can continue instead of restart.
# Lives next to the best-weights file. Delete it to force a fresh start.
CKPT_PATH = Path(str(C.WEIGHTS_BEST)).with_name("nnclr_last.ckpt")


# ---- augmentations ----
def _inject_bg(x):
    """Add a few faint Gaussian blobs as fake background sources."""
    B, _, H, W = x.shape
    dev = x.device
    n_src = int(torch.randint(0, C.AUG_BG_MAX_SRC + 1, (1,)).item())
    if n_src == 0:
        return x
    ys = torch.linspace(-1, 1, H, device=dev)
    xs = torch.linspace(-1, 1, W, device=dev)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    out = x.clone()
    for _ in range(n_src):
        cy = (torch.rand(B, device=dev) * 2 - 1)[:, None, None]
        cx = (torch.rand(B, device=dev) * 2 - 1)[:, None, None]
        sig = (0.02 + torch.rand(B, device=dev) * 0.05)[:, None, None]
        amp = (0.1 + torch.rand(B, device=dev) * 0.4)[:, None, None]
        blob = amp * torch.exp(-(((yy[None] - cy) ** 2 + (xx[None] - cx) ** 2)
                                 / (2 * sig ** 2)))
        out = out + blob.unsqueeze(1)
    return out


def augment(x):
    """Batched GPU galaxy augmentation. x: (B,1,H,W) in [0,1] -> one random view."""
    B = x.size(0)
    dev, dt = x.device, x.dtype

    if C.AUG_ROT_FULL:
        ang = (torch.rand(B, device=dev) * 2 - 1) * math.pi
    else:
        ang = (torch.rand(B, device=dev) * 2 - 1) * math.radians(C.AUG_ROT_SMALL_DEG)
    c, s = torch.cos(ang), torch.sin(ang)

    if C.AUG_FLIP:
        fx = torch.where(torch.rand(B, device=dev) < 0.5,
                         torch.full((B,), -1.0, device=dev), torch.ones(B, device=dev))
    else:
        fx = torch.ones(B, device=dev)

    tx = (torch.rand(B, device=dev) * 2 - 1) * C.AUG_TRANSLATE_FRAC
    ty = (torch.rand(B, device=dev) * 2 - 1) * C.AUG_TRANSLATE_FRAC

    mat = torch.zeros(B, 2, 3, device=dev, dtype=dt)
    mat[:, 0, 0] = c * fx; mat[:, 0, 1] = -s;  mat[:, 0, 2] = tx
    mat[:, 1, 0] = s * fx; mat[:, 1, 1] = c;   mat[:, 1, 2] = ty
    grid = F.affine_grid(mat, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros",
                      align_corners=False)

    if C.AUG_BG_INJECT:
        x = _inject_bg(x)

    lo, hi = C.AUG_CONTRAST
    con = torch.rand(B, 1, 1, 1, device=dev) * (hi - lo) + lo
    bri = (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * C.AUG_BRIGHTNESS
    x = x * con + bri
    if C.AUG_NOISE_STD > 0:
        x = x + torch.randn_like(x) * C.AUG_NOISE_STD
    return x.clamp(0.0, 1.0)


# ---- encoder (ResNet18) ----
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.sc = nn.Sequential()
        if stride != 1 or cin != cout:
            self.sc = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                    nn.BatchNorm2d(cout))

    def forward(self, x):
        o = F.relu(self.b1(self.c1(x)), inplace=True)
        o = self.b2(self.c2(o))
        return F.relu(o + self.sc(x), inplace=True)


class ResNetEncoder(nn.Module):
    """ResNet18 with a small-image stem (3x3 s1, no maxpool) -> 512-d feature."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 64, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.layer1 = self._stage(64, 64, 2, 1)
        self.layer2 = self._stage(64, 128, 2, 2)
        self.layer3 = self._stage(128, 256, 2, 2)
        self.layer4 = self._stage(256, 512, 2, 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 512

    @staticmethod
    def _stage(cin, cout, n, stride):
        blocks = [BasicBlock(cin, cout, stride)]
        for _ in range(n - 1):
            blocks.append(BasicBlock(cout, cout, 1))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return self.pool(x).flatten(1)


class ProjectionHead(nn.Module):
    def __init__(self, cin, hidden, out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cin, hidden), nn.BatchNorm1d(hidden),
                                 nn.ReLU(inplace=True), nn.Linear(hidden, out))

    def forward(self, x):
        return self.net(x)


# ---- NNCLR module ----
class NNCLR(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNetEncoder()
        self.proj = ProjectionHead(self.encoder.out_dim, C.PROJ_HIDDEN, C.PROJ_DIM)
        self.register_buffer("queue",
                             F.normalize(torch.randn(C.QUEUE_SIZE, C.PROJ_DIM), dim=1))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def embed(self, x):
        return F.normalize(self.proj(self.encoder(x)), dim=1)

    @torch.no_grad()
    def _enqueue(self, z):
        b = z.size(0)
        ptr = int(self.queue_ptr)
        if ptr + b <= C.QUEUE_SIZE:
            self.queue[ptr:ptr + b] = z
        else:
            first = C.QUEUE_SIZE - ptr
            self.queue[ptr:] = z[:first]
            self.queue[:b - first] = z[first:]
        self.queue_ptr[0] = (ptr + b) % C.QUEUE_SIZE

    @torch.no_grad()
    def _nn(self, z):
        """Nearest neighbour (cosine) of each row of z in the support queue."""
        idx = (z.float() @ self.queue.t()).argmax(dim=1)
        return self.queue[idx]

    def loss(self, z1, z2, update_queue=True):
        """NNCLR InfoNCE: positive = NN(view1) vs view2; negatives = other view2s.
        Done in fp32 for stability under AMP."""
        tau = C.TEMPERATURE
        z1f, z2f = z1.float(), z2.float()
        labels = torch.arange(z1f.size(0), device=z1f.device)
        logits = (self._nn(z1f) @ z2f.t()) / tau
        l = F.cross_entropy(logits, labels)
        if C.SYMMETRIC:
            logits2 = (self._nn(z2f) @ z1f.t()) / tau
            l = 0.5 * (l + F.cross_entropy(logits2, labels))
        if update_queue:
            self._enqueue(z1f.detach())
        return l


# ---- data staging + loading ----
def _stage_array(src_path, budget_gb):
    """RAM if it fits the budget (anonymous memory -> no eviction, no per-epoch
    time fluctuation); else node-local $TMPDIR memmap served by DataLoader workers."""
    src_path = str(src_path)
    mm = np.load(src_path, mmap_mode="r")
    gb = mm.size * mm.dtype.itemsize / 1e9
    if gb <= budget_gb:
        log.info(f"staging {gb:.1f} GB into RAM (one sequential read, then RAM-speed)")
        return np.array(mm), "ram"
    tmp = os.environ.get("TMPDIR", tempfile.gettempdir())
    dst = os.path.join(tmp, os.path.basename(src_path))
    if not (os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src_path)):
        log.info(f"staging {gb:.1f} GB to node-local {dst}")
        shutil.copy(src_path, dst)
    log.info(f"memmapping node-local copy ({gb:.1f} GB) with {C.NUM_WORKERS} workers")
    return np.load(dst, mmap_mode="r"), "memmap"


class _ImageDataset(Dataset):
    """Per-sample dataset, used ONLY for the memmap path (workers parallelise the
    random NVMe reads). The RAM path does not use this -- see _RamLoader."""
    def __init__(self, array, indices):
        self.array = array
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img = np.array(self.array[self.indices[i]], dtype=np.float32)  # copy -> writable
        return torch.from_numpy(img).unsqueeze(0)      # (1, H, W)


class _RamLoader:
    """Batches sliced directly from the in-RAM array and pinned for a
    non-blocking GPU copy. Avoids DataLoader workers, which would each copy the
    ~30 GB array. Reshuffles on every iteration."""

    def __init__(self, array, indices, batch, shuffle, seed, drop_last):
        self.array = array
        self.indices = np.asarray(indices)
        self.batch = batch
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)
        self.pin = (DEVICE.type == "cuda")

    def __len__(self):
        n = len(self.indices)
        return n // self.batch if self.drop_last else math.ceil(n / self.batch)

    def __iter__(self):
        idx = self.indices.copy()
        if self.shuffle:
            self.rng.shuffle(idx)
        n = len(idx)
        last = n - (n % self.batch) if self.drop_last else n
        for i in range(0, last, self.batch):
            b = idx[i:i + self.batch]
            chunk = np.ascontiguousarray(self.array[b], dtype=np.float32)
            x = torch.from_numpy(chunk).unsqueeze(1)        # (B, 1, H, W)
            yield x.pin_memory() if self.pin else x


def _make_loader(array, indices, mode, batch, shuffle, seed):
    if mode == "ram":
        return _RamLoader(array, indices, batch, shuffle, seed, drop_last=shuffle)
    return DataLoader(
        _ImageDataset(array, indices),
        batch_size=batch, shuffle=shuffle, num_workers=C.NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(C.NUM_WORKERS > 0), drop_last=shuffle,
        prefetch_factor=(4 if C.NUM_WORKERS > 0 else None))


def _to_model_dim(x):
    if x.shape[-1] != C.IMAGE_DIM:
        x = F.interpolate(x, size=(C.IMAGE_DIM, C.IMAGE_DIM), mode="area")
    return x


def _save_ckpt(path, model, opt, sched, scaler, epoch, best, no_imp, hist):
    """Atomic full-state save (write tmp, then os.replace) so a kill mid-write
    can't corrupt the checkpoint. model.state_dict() carries the support queue
    and queue_ptr buffers, so the queue resumes correctly too."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                "epoch": epoch, "best": best, "no_imp": no_imp, "hist": hist}, tmp)
    os.replace(tmp, path)


# ---- train ----
def train():
    torch.manual_seed(C.RANDOM_SEED)
    # fixed 128px input -> let cuDNN autotune conv algos once, then reuse them
    torch.backends.cudnn.benchmark = True
    use_amp = C.USE_AMP and DEVICE.type == "cuda"
    array, mode = _stage_array(C.IMAGES_NPY, C.RAM_BUDGET_GB)
    n = len(array)
    rng = np.random.default_rng(C.RANDOM_SEED)
    perm = rng.permutation(n)
    n_val = max(C.BATCH_SIZE, int(n * C.VAL_FRAC))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    log.info(f"stack {array.shape} -> dim {C.IMAGE_DIM} | mode {mode} | "
             f"train {len(train_idx):,} val {len(val_idx):,} | amp {use_amp}")

    tr_loader = _make_loader(array, train_idx, mode, C.BATCH_SIZE,
                             shuffle=True, seed=C.RANDOM_SEED)
    va_loader = _make_loader(array, val_idx, mode, C.BATCH_SIZE,
                             shuffle=False, seed=C.RANDOM_SEED + 1)

    model = NNCLR().to(DEVICE)
    log.info(f"NNCLR params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M "
             f"| rep_dim {C.REP_DIM} | proj {C.PROJ_DIM} | queue {C.QUEUE_SIZE} "
             f"| tau {C.TEMPERATURE}")
    opt = torch.optim.AdamW(model.parameters(), lr=C.LEARNING_RATE,
                            weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=C.N_EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # resume from the last full-state checkpoint if one exists
    best, no_imp, hist, start_epoch = float("inf"), 0, [], 1
    if CKPT_PATH.exists():
        ck = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        best, no_imp, hist = ck["best"], ck["no_imp"], ck["hist"]
        start_epoch = ck["epoch"] + 1
        log.info(f"resumed from {CKPT_PATH.name}: continuing at epoch "
                 f"{start_epoch}/{C.N_EPOCHS} (best val {best:.4f})")

    for ep in range(start_epoch, C.N_EPOCHS + 1):
        if ep <= C.WARMUP_EPOCHS:
            for g in opt.param_groups:
                g["lr"] = C.LEARNING_RATE * ep / C.WARMUP_EPOCHS

        model.train()
        t0 = time.time()
        tot = torch.zeros((), device=DEVICE); nb = 0     # accumulate on GPU, no per-step sync
        for batch in tr_loader:
            x = _to_model_dim(batch.to(DEVICE, non_blocking=True))
            v1, v2 = augment(x), augment(x)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = model.loss(model.embed(v1), model.embed(v2))
            scaler.scale(loss).backward()
            if C.GRAD_CLIP > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
            tot += loss.detach(); nb += 1
        if ep > C.WARMUP_EPOCHS:
            sched.step()
        train_loss = (tot / max(nb, 1)).item()           # single sync per epoch

        model.eval()
        vtot = torch.zeros((), device=DEVICE); vnb = 0
        with torch.no_grad():
            for batch in va_loader:
                x = _to_model_dim(batch.to(DEVICE, non_blocking=True))
                if x.shape[0] < 2:
                    continue
                with torch.amp.autocast("cuda", enabled=use_amp):
                    vtot += model.loss(model.embed(augment(x)),
                                       model.embed(augment(x)),
                                       update_queue=False).detach()
                vnb += 1
        val_loss = (vtot / max(vnb, 1)).item()

        dt = time.time() - t0
        log.info(f"epoch {ep:>3}/{C.N_EPOCHS} | train {train_loss:.4f} | "
                 f"val {val_loss:.4f} | {dt:.1f}s")
        hist.append({"epoch": ep, "train": train_loss, "val": val_loss})
        json.dump(hist, open(C.HISTORY_JSON, "w"))

        if val_loss < best - 1e-4:
            best, no_imp = val_loss, 0
            torch.save({"model": model.state_dict(), "epoch": ep, "val": val_loss},
                       C.WEIGHTS_BEST)
            log.info("  -> best, saved")
            improved = True
        else:
            no_imp += 1
            improved = False

        # full-state checkpoint every epoch so a TIMEOUT can resume from here
        _save_ckpt(CKPT_PATH, model, opt, sched, scaler, ep, best, no_imp, hist)

        if (not improved) and no_imp >= C.EARLY_STOP_PATIENCE and ep > C.WARMUP_EPOCHS:
            log.info(f"early stop at epoch {ep}")
            break
    log.info(f"done. best val {best:.4f}")


# ---- extract + PCA ----
@torch.no_grad()
def extract_features():
    ckpt = torch.load(C.WEIGHTS_BEST, map_location=DEVICE, weights_only=False)
    model = NNCLR().to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    array, _ = _stage_array(C.IMAGES_NPY, C.RAM_BUDGET_GB)
    n = len(array)
    labels = np.load(C.LABELS_NPY)
    feats = np.empty((n, C.REP_DIM), dtype=np.float32)
    bs = C.BATCH_SIZE
    for i in range(0, n, bs):
        chunk = np.array(array[i:i + bs], dtype=np.float32)
        t = _to_model_dim(torch.from_numpy(chunk).unsqueeze(1).to(DEVICE))
        feats[i:i + chunk.shape[0]] = model.encoder(t).cpu().numpy()
    np.save(C.LATENTS_NPY, feats)
    log.info(f"latents -> {C.LATENTS_NPY}  {feats.shape}")

    names = labels.dtype.names or ()
    train_mask = (labels["subfolder"] == "train") if "subfolder" in names \
        else np.ones(n, dtype=bool)
    k = min(C.N_COMPONENTS_MAX, C.REP_DIM)
    pca = PCA(n_components=k, random_state=C.RANDOM_SEED).fit(feats[train_mask])
    cum = np.cumsum(pca.explained_variance_ratio_)
    keep = min(max(int(np.searchsorted(cum, C.PCA_VAR_TARGET) + 1), 2), k)
    fp = pca.transform(feats)[:, :keep].astype(np.float32)
    np.save(C.FEATURES_NPY, fp)
    log.info(f"features_pca -> {C.FEATURES_NPY}  {fp.shape}  "
             f"(kept {keep}/{k}, cum var {cum[keep-1]:.4f})")


if __name__ == "__main__":
    train()
    extract_features()
