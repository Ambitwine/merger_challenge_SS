"""SimCLR arm: galaxy augmentations, a ResNet-18 encoder, NT-Xent training,
then feature extraction + PCA. Rotation invariance is learned from augmentation.

The image stack is read once into RAM when it fits C.RAM_BUDGET_GB (otherwise
copied to node-local $TMPDIR and memmapped), so training is not limited by
random reads from shared /scratch.

    python -u simclr_arm.py      # train, then extract + PCA
"""
from __future__ import annotations
import json
import logging
import math
import os
import shutil
import tempfile
import time

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset

import config_simclr as C

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---- augmentations ----
class GalaxyAugment:
    """Batched, on-GPU galaxy augmentations. Each call returns ONE augmented
    view; call twice for the two SimCLR views."""

    def __init__(self, image_dim=C.IMAGE_DIM, rotation=C.AUG_ROTATION,
                 flip=C.AUG_FLIP, translate_frac=C.AUG_TRANSLATE_FRAC,
                 noise_sigma=C.AUG_NOISE_SIGMA, contrast=C.AUG_CONTRAST,
                 brightness=C.AUG_BRIGHTNESS, bg_sources=C.AUG_BG_SOURCES,
                 bg_flux=C.AUG_BG_FLUX):
        self.H = image_dim
        self.rotation = rotation
        self.flip = flip
        self.tfrac = translate_frac
        self.noise = noise_sigma
        self.contrast = contrast
        self.brightness = brightness
        self.bg_sources = bg_sources
        self.bg_flux = bg_flux

    def _geom(self, x):
        B = x.shape[0]
        dev, dt = x.device, x.dtype
        a = (torch.rand(B, device=dev, dtype=dt) * 2 * math.pi
             if self.rotation else torch.zeros(B, device=dev, dtype=dt))
        c, s = torch.cos(a), torch.sin(a)
        sx = torch.ones(B, device=dev, dtype=dt)
        sy = torch.ones(B, device=dev, dtype=dt)
        if self.flip:
            sx = torch.where(torch.rand(B, device=dev) < 0.5, -sx, sx)
            sy = torch.where(torch.rand(B, device=dev) < 0.5, -sy, sy)
        tx = (torch.rand(B, device=dev, dtype=dt) * 2 - 1) * self.tfrac
        ty = (torch.rand(B, device=dev, dtype=dt) * 2 - 1) * self.tfrac
        mat = torch.zeros(B, 2, 3, device=dev, dtype=dt)
        mat[:, 0, 0] = sx * c;  mat[:, 0, 1] = -sx * s; mat[:, 0, 2] = tx
        mat[:, 1, 0] = sy * s;  mat[:, 1, 1] = sy * c;  mat[:, 1, 2] = ty
        grid = F.affine_grid(mat, x.shape, align_corners=False)
        return F.grid_sample(x, grid, mode="bilinear",
                             padding_mode="zeros", align_corners=False)

    def _inject_bg(self, x):
        B, _, H, W = x.shape
        dev, dt = x.device, x.dtype
        ys = torch.arange(H, device=dev, dtype=dt)
        xs = torch.arange(W, device=dev, dtype=dt)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        for _ in range(self.bg_sources):
            cyx = torch.rand(B, 2, device=dev, dtype=dt) * (H - 1)
            flux = (torch.rand(B, device=dev, dtype=dt)
                    * (self.bg_flux[1] - self.bg_flux[0]) + self.bg_flux[0])
            sig = torch.rand(B, device=dev, dtype=dt) * 1.0 + 0.8
            d2 = ((xx[None] - cyx[:, 1, None, None]) ** 2
                  + (yy[None] - cyx[:, 0, None, None]) ** 2)
            blob = flux[:, None, None] * torch.exp(-d2 / (2 * sig[:, None, None] ** 2))
            x = x + blob.unsqueeze(1)
        return x

    @torch.no_grad()
    def __call__(self, x):
        x = self._geom(x)
        if self.contrast is not None:
            lo, hi = self.contrast
            cf = torch.rand(x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype) * (hi - lo) + lo
            x = x * cf
        if self.brightness:
            bf = (torch.rand(x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype) * 2 - 1) * self.brightness
            x = x + bf
        if self.bg_sources > 0:
            x = self._inject_bg(x)
        if self.noise:
            x = x + torch.randn_like(x) * self.noise
        return x.clamp(0.0, 1.0)


# ---- encoder (ResNet-18) ----
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.short = nn.Sequential()
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)), inplace=True)
        h = self.bn2(self.conv2(h))
        return F.relu(h + self.short(x), inplace=True)


class ResNetEncoder(nn.Module):
    """Compact ResNet-18 with a small-image stem (3x3 s1, no max-pool).
    forward(x) -> (B, REP_DIM) global-average-pooled features."""

    def __init__(self, in_ch=C.IN_CHANNELS, width=C.RESNET_WIDTH,
                 layers=C.RESNET_LAYERS):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.in_c = width
        self.layer1 = self._make(width,      layers[0], 1)
        self.layer2 = self._make(width * 2,  layers[1], 2)
        self.layer3 = self._make(width * 4,  layers[2], 2)
        self.layer4 = self._make(width * 8,  layers[3], 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.rep_dim = width * 8
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def _make(self, cout, n, stride):
        blocks = [BasicBlock(self.in_c, cout, stride)]
        self.in_c = cout
        for _ in range(1, n):
            blocks.append(BasicBlock(cout, cout, 1))
        return nn.Sequential(*blocks)

    def forward(self, x):
        h = self.stem(x)
        h = self.layer4(self.layer3(self.layer2(self.layer1(h))))
        return self.pool(h).flatten(1)


class ProjectionHead(nn.Module):
    def __init__(self, rep_dim=C.REP_DIM, hidden=C.PROJ_HIDDEN, out=C.PROJ_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(rep_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out))

    def forward(self, x):
        return self.net(x)


class SimCLR(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNetEncoder()
        self.head = ProjectionHead(rep_dim=self.encoder.rep_dim)

    def forward(self, x):
        return self.head(self.encoder(x))      # projection (for the loss)

    @torch.no_grad()
    def represent(self, x):
        return self.encoder(x)                 # backbone features (for clustering)


# ---- NT-Xent loss ----
def nt_xent(z1, z2, temp=C.NT_XENT_TEMP):
    """NT-Xent over the two views z1, z2 (B, D). Computed in fp32 because under
    AMP the projections arrive as fp16. Positives are (i, i); the other 2B-2
    entries in each row are negatives."""
    B = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2], 0).float(), dim=1)     # (2B, D), fp32
    sim = (z @ z.T) / temp                                     # (2B, 2B)
    sim.fill_diagonal_(float("-inf"))                          # mask self-similarity
    targets = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, targets)


# ---- data staging + loading ----
def _load_labels_and_split():
    labels = np.load(C.LABELS_NPY)
    names = labels.dtype.names or ()
    if "subfolder" in names:
        train_mask = (labels["subfolder"] == "train")
    else:
        rng = np.random.default_rng(C.RANDOM_SEED)
        train_mask = rng.random(len(labels)) >= C.VAL_FRAC_FALLBACK
    return labels, train_mask


def _stage_array(src_path, budget_gb):
    """Return (array, mode). RAM if it fits the budget (anonymous memory -> no
    page-cache eviction, no per-epoch time fluctuation); else a node-local
    $TMPDIR memmap (local NVMe random reads, served by DataLoader workers)."""
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
        self.pin = (_DEVICE.type == "cuda")

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
        pin_memory=(_DEVICE.type == "cuda"),
        persistent_workers=(C.NUM_WORKERS > 0), drop_last=shuffle,
        prefetch_factor=(4 if C.NUM_WORKERS > 0 else None))


def _to_model_dim(x):
    if x.shape[-1] != C.IMAGE_DIM:
        x = F.interpolate(x, size=(C.IMAGE_DIM, C.IMAGE_DIM), mode="area")
    return x


def _save_ckpt(model, opt, scaler, epoch, gstep, best_val, bad, history):
    """Atomic full-state save (write tmp, then os.replace) so a kill mid-write
    can't corrupt the checkpoint and a TIMEOUT can resume from the last epoch."""
    tmp = C.CKPT_PATH.with_suffix(C.CKPT_PATH.suffix + ".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "scaler": scaler.state_dict(), "epoch": epoch, "gstep": gstep,
                "best_val": best_val, "bad": bad, "history": history}, tmp)
    os.replace(tmp, C.CKPT_PATH)


# ---- training ----
def train():
    torch.manual_seed(C.RANDOM_SEED)
    np.random.seed(C.RANDOM_SEED)
    torch.backends.cudnn.benchmark = True   # fixed 128px input -> autotune once
    labels, train_mask = _load_labels_and_split()
    array, mode = _stage_array(C.IMAGES_NPY, C.RAM_BUDGET_GB)
    if len(array) != len(labels):
        raise RuntimeError(f"stack/labels mismatch: {len(array)} vs {len(labels)}")
    idx_tr = np.where(train_mask)[0]
    idx_va = np.where(~train_mask)[0]
    log.info(f"device={_DEVICE} | mode={mode} | train={len(idx_tr):,} "
             f"val={len(idx_va):,} | native={array.shape[1]}px -> model {C.IMAGE_DIM}px")

    tr_loader = _make_loader(array, idx_tr, mode, C.BATCH_SIZE,
                             shuffle=True, seed=C.RANDOM_SEED)
    va_loader = _make_loader(array, idx_va, mode, C.BATCH_SIZE,
                             shuffle=False, seed=C.RANDOM_SEED + 1)

    model = SimCLR().to(_DEVICE)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"SimCLR params: {n_par/1e6:.2f}M (encoder rep_dim={model.encoder.rep_dim})")
    aug = GalaxyAugment()
    opt = torch.optim.AdamW(model.parameters(), lr=C.LEARNING_RATE,
                            weight_decay=C.WEIGHT_DECAY)
    steps_per_epoch = max(1, len(tr_loader))
    total_steps = C.N_EPOCHS * steps_per_epoch
    warmup = C.WARMUP_EPOCHS * steps_per_epoch

    def lr_at(step):
        if step < warmup:
            return step / max(1, warmup)
        p = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))

    use_amp = C.USE_AMP and _DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history = {"train_loss": [], "val_loss": []}
    best_val, bad, gstep, start_epoch = float("inf"), 0, 0, 1

    # resume from the last full-state checkpoint if one exists
    if C.CKPT_PATH.exists():
        ck = torch.load(C.CKPT_PATH, map_location=_DEVICE, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        gstep, best_val, bad = ck["gstep"], ck["best_val"], ck["bad"]
        history = ck["history"]
        start_epoch = ck["epoch"] + 1
        log.info(f"resumed from {os.path.basename(str(C.CKPT_PATH))}: continuing at "
                 f"epoch {start_epoch}/{C.N_EPOCHS} (gstep {gstep}, best val {best_val:.4f})")

    for ep in range(start_epoch, C.N_EPOCHS + 1):
        model.train(); t0 = time.time()
        tl = torch.zeros((), device=_DEVICE); tn = 0     # GPU accum, one sync/epoch
        for batch in tr_loader:
            x = _to_model_dim(batch.to(_DEVICE, non_blocking=True))
            for g in opt.param_groups:
                g["lr"] = C.LEARNING_RATE * lr_at(gstep)
            v1, v2 = aug(x), aug(x)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = nt_xent(model(v1), model(v2))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            scaler.step(opt); scaler.update()
            tl += loss.detach() * x.shape[0]; tn += x.shape[0]; gstep += 1
        train_loss = (tl / max(tn, 1)).item()

        model.eval()
        vl = torch.zeros((), device=_DEVICE); vn = 0
        with torch.no_grad():
            for batch in va_loader:
                x = _to_model_dim(batch.to(_DEVICE, non_blocking=True))
                if x.shape[0] < 2:
                    continue
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss = nt_xent(model(aug(x)), model(aug(x)))
                vl += loss.detach() * x.shape[0]; vn += x.shape[0]
        val_loss = (vl / max(vn, 1)).item()
        history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)
        log.info(f"epoch {ep:>3}/{C.N_EPOCHS} | train {train_loss:.4f} | "
                 f"val {val_loss:.4f} | {time.time()-t0:.1f}s")

        if val_loss < best_val - 1e-4:
            best_val, bad = val_loss, 0
            torch.save({"encoder": model.encoder.state_dict(),
                        "rep_dim": model.encoder.rep_dim, "epoch": ep,
                        "val_loss": val_loss, "image_dim": C.IMAGE_DIM},
                       C.BEST_ENCODER)
            stop = False
        else:
            bad += 1
            stop = bad >= C.EARLY_STOP_PATIENCE

        # full-state checkpoint every epoch (atomic) so a TIMEOUT can resume
        _save_ckpt(model, opt, scaler, ep, gstep, best_val, bad, history)
        C.HISTORY_JSON.write_text(json.dumps(history, indent=2))

        if stop:
            log.info(f"early stop at epoch {ep} (best val {best_val:.4f})")
            break
    log.info(f"done. best encoder -> {C.BEST_ENCODER}")


# ---- feature extraction ----
@torch.no_grad()
def extract_features(weights=None):
    """Encode all images with the trained backbone, then PCA fit on train rows
    only, to PCA_VARIANCE_TARGET."""
    weights = weights or C.BEST_ENCODER
    ckpt = torch.load(weights, map_location=_DEVICE, weights_only=False)
    enc = ResNetEncoder().to(_DEVICE).eval()
    enc.load_state_dict(ckpt["encoder"])
    log.info(f"loaded encoder from {weights} (epoch {ckpt.get('epoch')}, "
             f"val {ckpt.get('val_loss'):.4f})")

    labels, train_mask = _load_labels_and_split()
    array, _ = _stage_array(C.IMAGES_NPY, C.RAM_BUDGET_GB)
    n = len(array)
    reps = np.empty((n, enc.rep_dim), dtype=np.float32)
    bs = C.BATCH_SIZE
    for i in range(0, n, bs):
        chunk = np.array(array[i:i + bs], dtype=np.float32)
        t = _to_model_dim(torch.from_numpy(chunk).unsqueeze(1).to(_DEVICE))
        reps[i:i + chunk.shape[0]] = enc(t).cpu().numpy()
        if (i // bs) % 50 == 0:
            log.info(f"  encoded {min(i + bs, n):,}/{n:,}")
    np.save(C.LATENTS_NPY, reps)
    log.info(f"latents -> {C.LATENTS_NPY}: {reps.shape}")

    tr = reps[train_mask]
    pca_full = PCA(n_components=min(tr.shape)).fit(tr)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    n_keep = min(int(np.searchsorted(cum, C.PCA_VARIANCE_TARGET) + 1), C.N_COMPONENTS_MAX)
    pca = PCA(n_components=n_keep).fit(tr)
    joblib.dump(pca, C.PCA_JOBLIB)
    feats = pca.transform(reps).astype(np.float32)
    np.save(C.FEATURES_NPY, feats)
    log.info(f"PCA: {n_keep} comps for {C.PCA_VARIANCE_TARGET:.3%} var "
             f"(capped at {C.N_COMPONENTS_MAX}) -> {C.FEATURES_NPY}: {feats.shape}")
    return feats


if __name__ == "__main__":
    train()
    extract_features()
