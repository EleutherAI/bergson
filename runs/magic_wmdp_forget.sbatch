#!/bin/bash
#SBATCH --job-name=magic-forget
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=/home/a6a/lucia.a6a/bergson3/runs/magic-forget-%j.out

module load PrgEnv-cray
module load cuda/12.6

export HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_HOME=/home/a6a/lucia.a6a/.cache/torch
export TORCHINDUCTOR_CACHE_DIR=/home/a6a/lucia.a6a/.cache/torch_inductor

cd /home/a6a/lucia.a6a/bergson3
source .venv/bin/activate

python -u runs/magic_per_token.py --dataset wmdp_forget
