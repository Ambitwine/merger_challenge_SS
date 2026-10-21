"""Process the Horizon-AGN or HSC images into TNG-matched 128 px arcsinh [0,1]
stacks plus label arrays, for the domain-transfer test.

The image half (resize + arcsinh normalisation) is identical to the TNG
preprocessing, so the TNG-trained encoder sees in-distribution inputs. Only the
catalogue/label half differs per dataset. These catalogues have no snapnum
column to build file paths from, so each image is matched to its catalogue row
from the filename (see build_manifest).

Run on Kapteyn, where the raw data lives:
    python prepare_transfer_data.py --dataset horizon --extract --build --workers 8
    python prepare_transfer_data.py --dataset hsc     --extract --build --workers 8
"""
from __future__ import annotations
import argparse
import logging
import multiprocessing as mp
import re
import tarfile
import time
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# normalisation recipe, same values as the TNG preprocessing
NORM_ARCSINH_A     = 0.1
NORM_CLIP_LOW_PCT  = 0.5
NORM_CLIP_HIGH_PCT = 99.5
CUTOUT_SIZE_PX     = 128

ROOT = Path("/net/virgo01/data/users/risueno")
MC   = ROOT / "merger_challenge"
DATASETS = {
    "horizon": {
        "tarball_dir": MC / "AGN",                       # horizon_hsc_snapnum_*.tar.gz
        "extracted":   MC / "AGN" / "extracted",
        "catalog":     MC / "catalog_Horizon_AGN.fits",
        "out_images":  ROOT / "processed_horizon" / "images_horizon.npy",
        "out_labels":  ROOT / "processed_horizon" / "labels_horizon.npy",
    },
    "hsc": {
        "tarball_dir": MC / "HSC",                        # data_hsc_snapnum_*.tar.gz
        "extracted":   MC / "HSC" / "extracted",
        "catalog":     MC / "catalog_merger_challenge_HSC.fits",
        "out_images":  ROOT / "processed_hsc" / "images_hsc.npy",
        "out_labels":  ROOT / "processed_hsc" / "labels_hsc.npy",
    },
}


def extract_tarballs(tarball_dir: Path, extracted: Path, force: bool = False) -> None:
    extracted.mkdir(parents=True, exist_ok=True)
    tarballs = sorted(tarball_dir.glob("*.tar.gz"))
    if not tarballs:
        log.warning(f"no *.tar.gz found in {tarball_dir}")
    for src in tarballs:
        stem = src.name.replace(".tar.gz", "")
        marker = extracted / stem
        if not force and marker.exists() and sum(1 for _ in marker.rglob("*.fits")) > 100:
            log.info(f"  SKIP (already extracted): {src.name}")
            continue
        log.info(f"  Extracting {src.name} ({src.stat().st_size/1e9:.1f} GB) ...")
        t0 = time.time()
        with tarfile.open(src, "r:gz") as tf:
            tf.extractall(extracted, filter="data")
        log.info(f"    done in {(time.time()-t0)/60:.1f} min")
    log.info("Extraction complete.")


def build_manifest(extracted: Path, cat: Table) -> np.ndarray:
    """Map each catalog row to one image file.

    Filenames look like 'horizon_<proj>_ojbID_<NUM>' (projection prefix, and a
    'ojbID' typo), and <NUM> is some catalog id column -- not necessarily 'ID'.
    So: parse the trailing integer from each filename, keep ONE projection per
    object (prefer xy) so the stack is 1:1 with the labels, then auto-detect
    which catalog integer column those numbers index by maximum overlap."""
    log.info(f"Scanning {extracted} for *.fits ...")
    files = list(extracted.rglob("*.fits"))
    log.info(f"  found {len(files):,} fits files; sample stems {[p.stem for p in files[:3]]}")
    if not files:
        log.error("no .fits found -- did extraction run? is tarball_dir correct?")
        return np.array([""] * len(cat), dtype=object)

    PROJ_ORDER = ["xy", "yz", "xz"]
    num_re  = re.compile(r"(\d+)\D*$")          # trailing integer in the stem
    proj_re = re.compile(r"_(xy|yz|xz)_")
    by_num = {}                                  # object number -> (proj_rank, path)
    for p in files:
        m = num_re.search(p.stem)
        if not m:
            continue
        num = int(m.group(1))
        pm = proj_re.search(p.stem)
        rank = PROJ_ORDER.index(pm.group(1)) if pm else len(PROJ_ORDER)
        if num not in by_num or rank < by_num[num][0]:
            by_num[num] = (rank, str(p))
    fits_nums = list(by_num)
    log.info(f"  parsed {len(fits_nums):,} unique object numbers (one projection each)")

    best = None
    for c in cat.colnames:
        col = np.asarray(cat[c])
        if not np.issubdtype(col.dtype, np.integer):
            continue
        vals = col.astype(np.int64)
        overlap = int(np.isin(vals, fits_nums).sum())
        log.info(f"    column '{c}': {overlap:,}/{len(cat):,} rows match a fits number")
        if best is None or overlap > best[1]:
            best = (c, overlap, vals)
    if best is None or best[1] == 0:
        log.error(f"no integer column matches. sample fits numbers: {fits_nums[:10]}")
        return np.array([""] * len(cat), dtype=object)
    col, hits, vals = best
    log.info(f"  -> matching on catalog column '{col}' ({hits:,}/{len(cat):,} matched)")
    return np.array([by_num.get(int(v), (0, ""))[1] for v in vals], dtype=object)


def normalize_image(img: np.ndarray) -> np.ndarray:
    a = NORM_ARCSINH_A
    img = np.arcsinh(img / a) / np.arcsinh(1.0 / a)
    lo = np.percentile(img, NORM_CLIP_LOW_PCT)
    hi = np.percentile(img, NORM_CLIP_HIGH_PCT)
    img = np.clip(img, lo, hi)
    denom = hi - lo
    img = (img - lo) / denom if denom > 0 else np.zeros_like(img)
    return img.astype(np.float32)


def _resize_to_square(img: np.ndarray, target: int) -> np.ndarray:
    h, w = img.shape
    if h != w:
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        img = img[y0:y0 + side, x0:x0 + side]
        h = w = side
    if h == target:
        return img
    if h % target == 0:
        block = h // target
        img = img.reshape(target, block, target, block).mean(axis=(1, 3))
    else:
        from scipy.ndimage import zoom
        img = zoom(img, target / h, order=1)
        if img.shape[0] != target:
            img = img[:target, :target]
    return img.astype(np.float32)


def _load_one(args):
    idx, path = args
    if not path:
        return idx, None
    try:
        with fits.open(path, memmap=False) as hdul:
            data = hdul[0].data
        if data is None:
            return idx, None
        data = _resize_to_square(data.astype(np.float32), CUTOUT_SIZE_PX)
        return idx, normalize_image(data)
    except Exception as e:
        log.warning(f"failed to load {path}: {e}")
        return idx, None


def build_labels_horizon(cat: Table) -> np.ndarray:
    n = len(cat)
    dt = np.asarray(cat["dt"], dtype=float)
    mm = np.asarray(cat["is_major_merger"]).astype(int)
    pre  = ((mm == 1) & (dt >= -0.8) & (dt < -0.1)).astype(np.int8)
    ong  = ((mm == 1) & (dt >= -0.1) & (dt <= 0.1)).astype(np.int8)
    post = ((mm == 1) & (dt > 0.1) & (dt <= 0.3)).astype(np.int8)
    tree = np.asarray(cat["tree_id"]).astype(np.int64)
    ids  = np.asarray(cat["ID"]).astype(np.int64)
    seq  = np.where(tree > 0, tree, -(ids + 1)).astype(np.int64)
    dtype = np.dtype([
        ("ID", "i8"), ("sim_ID", "U20"), ("redshift", "f4"),
        ("is_major_merger", "i1"), ("is_pre_merger", "i1"),
        ("is_ongoing_merger", "i1"), ("is_post_merger", "i1"),
        ("sequence_ID", "i8"), ("dt", "f4"), ("tree_id", "i8"),
        ("data_type", "U8"), ("subfolder", "U5"),
    ])
    out = np.empty(n, dtype=dtype)
    out["ID"] = ids
    out["sim_ID"] = [s.decode() if isinstance(s, bytes) else s for s in cat["sim_ID"]]
    out["redshift"] = cat["redshift"]
    out["is_major_merger"] = mm
    out["is_pre_merger"], out["is_ongoing_merger"], out["is_post_merger"] = pre, ong, post
    out["sequence_ID"], out["dt"], out["tree_id"] = seq, dt, tree
    out["data_type"], out["subfolder"] = "horizon", ""
    log.info(f"  horizon labels: merger {int((mm==1).sum()):,} non {int((mm==0).sum()):,} "
             f"| pre/ong/post {int(pre.sum()):,}/{int(ong.sum()):,}/{int(post.sum()):,}")
    return out


def build_labels_hsc(cat: Table) -> np.ndarray:
    n = len(cat)
    mm  = np.asarray(cat["is_major_merger"], dtype=float)
    dis = np.asarray(cat["is_disturbed"], dtype=float)
    mm_i  = np.where(mm  < 0, -1, mm).astype(np.int8)
    dis_i = np.where(dis < 0, -1, dis).astype(np.int8)
    labeled = (mm_i >= 0).astype(np.int8)
    dtype = np.dtype([
        ("ID", "i8"), ("redshift", "f4"),
        ("is_major_merger", "i1"), ("is_disturbed", "i1"), ("labeled", "i1"),
        ("data_type", "U8"), ("subfolder", "U5"),
    ])
    out = np.empty(n, dtype=dtype)
    out["ID"] = np.asarray(cat["ID"]).astype(np.int64)
    out["redshift"] = cat["redshift"]
    out["is_major_merger"], out["is_disturbed"], out["labeled"] = mm_i, dis_i, labeled
    out["data_type"], out["subfolder"] = "hsc", ""
    log.info(f"  hsc labels: labeled {int(labeled.sum()):,} "
             f"(merger {int((mm_i==1).sum()):,} non {int((mm_i==0).sum()):,}) "
             f"unlabeled {int((labeled==0).sum()):,}")
    return out


LABEL_BUILDERS = {"horizon": build_labels_horizon, "hsc": build_labels_hsc}


def build_stack(dataset, paths, cat, out_images: Path, out_labels: Path, num_workers=8) -> None:
    n_total = len(paths)
    H = CUTOUT_SIZE_PX
    out_images.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Building {dataset} stack: {n_total:,} entries, {num_workers} workers -> {out_images}")
    tmp = out_images.with_suffix(".tmp.npy")
    stack = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(n_total, H, H))
    success = np.zeros(n_total, dtype=bool)
    t0, done = time.time(), 0
    with mp.Pool(num_workers) as pool:
        for idx, img in pool.imap_unordered(_load_one, [(i, paths[i]) for i in range(n_total)], chunksize=64):
            done += 1
            if img is not None:
                stack[idx] = img
                success[idx] = True
            if done % 10000 == 0:
                rate = done / (time.time() - t0)
                log.info(f"  {done:,}/{n_total:,} ({success[:done].sum():,} OK) "
                         f"@ {rate:.0f}/s ETA {(n_total-done)/rate/60:.0f} min")
    stack.flush(); del stack
    log.info(f"Loaded. success {success.sum():,}/{n_total:,}")

    # keep only rows with an image, so the stack and labels stay row-aligned
    keep = np.where(success)[0]
    src = np.load(tmp, mmap_mode="r")
    final = np.lib.format.open_memmap(out_images, mode="w+", dtype=np.float32, shape=(len(keep), H, H))
    for j, i in enumerate(keep):
        final[j] = src[i]
    final.flush(); del final, src
    tmp.unlink()
    log.info(f"  wrote {out_images} ({len(keep):,}, {H}, {H})")

    labels = LABEL_BUILDERS[dataset](cat[keep])
    np.save(out_labels, labels)
    log.info(f"  wrote {out_labels}. DONE.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--force-extract", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if not (args.extract or args.build):
        ap.error("specify --extract and/or --build")
    cfg = DATASETS[args.dataset]
    if args.extract:
        extract_tarballs(cfg["tarball_dir"], cfg["extracted"], force=args.force_extract)
    if args.build:
        cat = Table.read(cfg["catalog"])
        log.info(f"catalog {cfg['catalog'].name}: {len(cat):,} rows, cols {cat.colnames}")
        paths = build_manifest(cfg["extracted"], cat)
        if int((paths != "").sum()) == 0:
            log.error("0 files matched the catalog -- not building. Fix matching first.")
            return
        build_stack(args.dataset, paths, cat, cfg["out_images"], cfg["out_labels"], args.workers)


if __name__ == "__main__":
    main()
