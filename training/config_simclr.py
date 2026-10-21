"""SimCLR arm config: paths, model dims, augmentation and training settings.

Reads the shared TNG image stack and labels and writes to its own model and
results directories.
"""
from __future__ import annotations
import os
from pathlib import Path

DATA_ROOT     = Path(os.environ.get("SIMCLR_DATA_ROOT", "/scratch/s4743091/tng"))
PROCESSED_DIR = DATA_ROOT / "processed_tng"
IMAGES_NPY    = PROCESSED_DIR / "images_tng.npy"   # (N, 128, 128) float32 in [0,1]
LABELS_NPY    = PROCESSED_DIR / "labels_tng.npy"   # structured: is_major_merger, subfolder, ...

MODELS_DIR  = DATA_ROOT / "models" / "simclr_tng"
RESULTS_DIR = DATA_ROOT / "results_tng" / "simclr"
for d in (MODELS_DIR, RESULTS_DIR):
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

# ---- Image geometry ---------------------------------------------------------
SOURCE_SIZE_PX = 128
IMAGE_DIM      = 128
IN_CHANNELS    = 1

# ---- Encoder / projection ---------------------------------------------------
RESNET_LAYERS = (2, 2, 2, 2)     # resnet-18
RESNET_WIDTH  = 64               # base width -> REP_DIM = 8 * WIDTH = 512
REP_DIM       = 8 * RESNET_WIDTH # 512  (encoder feature dim used for clustering)
PROJ_HIDDEN   = 512
PROJ_DIM      = 128              # contrastive embedding dim (discarded after training)

# ---- Contrastive loss -------------------------------------------------------
NT_XENT_TEMP = 0.2               # SimCLR temperature

# ---- Augmentations ----------------------------------------------------------
AUG_ROTATION       = True
AUG_FLIP           = True
AUG_TRANSLATE_FRAC = 0.06
AUG_NOISE_SIGMA    = 0.02
AUG_CONTRAST       = (0.85, 1.15)
AUG_BRIGHTNESS     = 0.03
AUG_BG_SOURCES     = 0
AUG_BG_FLUX        = (0.05, 0.3)

# ---- Training ---------------------------------------------------------------
BATCH_SIZE          = 512
N_EPOCHS            = 300
LEARNING_RATE       = 3e-3
WEIGHT_DECAY        = 1e-4
WARMUP_EPOCHS       = 10
EARLY_STOP_PATIENCE = 30
RANDOM_SEED         = 42
USE_AMP             = True
GRAD_CLIP           = 5.0
NUM_WORKERS         = 8
VAL_FRAC_FALLBACK   = 0.1

# ---- Data staging ----
# Stacks up to RAM_BUDGET_GB are read once into RAM; larger ones are copied to
# node-local $TMPDIR and memmapped. Raise together with --mem in the SLURM script.
RAM_BUDGET_GB = 40

# ---- PCA (+ BGM settings from an earlier clustering version) ----
PCA_VARIANCE_TARGET = 0.999
N_COMPONENTS_MAX    = 25
BGM_WEIGHT_PRIOR    = 1e-2
BGM_MAX_ITER        = 500
BGM_N_INIT          = 3

# ---- Output filenames -------------------------------------------------------
BEST_ENCODER  = MODELS_DIR / "simclr_encoder_best.pt"
CKPT_PATH     = MODELS_DIR / "simclr_tng.ckpt"
HISTORY_JSON  = MODELS_DIR / "training_history.json"
LATENTS_NPY   = MODELS_DIR / "latents_simclr.npy"
PCA_JOBLIB    = MODELS_DIR / "pca_simclr.joblib"
FEATURES_NPY  = MODELS_DIR / "features_pca_simclr.npy"
