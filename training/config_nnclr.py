"""NNCLR arm config (Dwibedi et al. 2021). Mirrors config_simclr.py, with the
NNCLR-specific settings (support-set queue, temperature) and separate output
paths so both arms can run at the same time.
"""
from __future__ import annotations
import os
from pathlib import Path

# NNCLR_DATA_ROOT lets a test box override the cluster path; default is Habrok.
DATA_ROOT     = Path(os.environ.get("NNCLR_DATA_ROOT", "/scratch/s4743091/tng"))
PROCESSED_DIR = DATA_ROOT / "processed_tng"
IMAGES_NPY    = PROCESSED_DIR / "images_tng.npy"      # native 128px stack
LABELS_NPY    = PROCESSED_DIR / "labels_tng.npy"

MODELS_DIR    = DATA_ROOT / "models" / "nnclr_tng"
RESULTS_DIR   = DATA_ROOT / "results_tng" / "nnclr"
for _d in (MODELS_DIR, RESULTS_DIR):
    try:
        _d.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass  # read-only / non-existent root on a test box -- harmless

LATENTS_NPY   = MODELS_DIR / "latents_nnclr.npy"
FEATURES_NPY  = MODELS_DIR / "features_pca_nnclr.npy"
WEIGHTS_BEST  = MODELS_DIR / "nnclr_best.pt"
HISTORY_JSON  = MODELS_DIR / "training_history_nnclr.json"

# ---- data / model ----------------------------------------------------------
IMAGE_DIM    = 128      # resize-on-load if the stored stack differs
REP_DIM      = 512      # ResNet18 backbone output -- the features we keep
PROJ_HIDDEN  = 512
PROJ_DIM     = 256      # projection / contrast space (paper rep dim = 256)

# ---- NNCLR -----------------------------------------------------------------
TEMPERATURE  = 0.1      # NNCLR default (SimCLR used 0.2)
QUEUE_SIZE   = 32768    # support set; ~7% of the 463k sample -- plenty of NNs
SYMMETRIC    = True     # symmetrise the loss over the two views

# ---- training --------------------------------------------------------------
BATCH_SIZE          = 512
N_EPOCHS            = 300
LEARNING_RATE       = 1e-3
WEIGHT_DECAY        = 1e-6
WARMUP_EPOCHS       = 10
EARLY_STOP_PATIENCE = 30
VAL_FRAC            = 0.05
GRAD_CLIP           = 5.0
USE_AMP             = True
RANDOM_SEED         = 0
NUM_WORKERS         = 8

# ---- data staging (same scheme as config_simclr) ----
RAM_BUDGET_GB = 40

# ---- PCA ----
PCA_VAR_TARGET   = 0.999
N_COMPONENTS_MAX = 25

# ---- augmentations ---------------------------------------------------------
# Full 360-degree rotation: with a plain ResNet, rotation invariance has to come
# from the data. Small rotations only make sense with an equivariant backbone.
AUG_ROT_FULL       = True
AUG_ROT_SMALL_DEG  = 10.0          # paper's value; ONLY sensible with steerable convs
AUG_FLIP           = True
AUG_TRANSLATE_FRAC = 0.06          # ~+/-8 px at 128
AUG_CONTRAST       = (0.85, 1.15)
AUG_BRIGHTNESS     = 0.05
AUG_NOISE_STD      = 0.02
AUG_BG_INJECT      = False         # optional fake interlopers (companions robustness)
AUG_BG_MAX_SRC     = 4
