#!/bin/bash
#SBATCH --job-name=simclr_tng
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=simclr_%j.log

module purge
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
module load scikit-learn/1.6.1-gfbf-2024a

echo "host: $(hostname)   job: $SLURM_JOB_ID   $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

cd ~/Thesis/TNG/simclr || { echo "FATAL: cd failed"; exit 1; }

# ---- preflight: fail loudly here (not 40s into training) if data is missing ----
python -c "import config_simclr as C, os, numpy as np; p=str(C.IMAGES_NPY); print('stack :', p, '->', os.path.exists(p)); print('labels:', str(C.LABELS_NPY), '->', os.path.exists(str(C.LABELS_NPY))); print('shape :', np.load(p, mmap_mode='r').shape if os.path.exists(p) else 'MISSING')"

python -u simclr_arm.py
