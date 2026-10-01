#!/bin/bash
#SBATCH --job-name=compile_grad_probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:15:00
#SBATCH --output=logs/compile_grad_probe_%A.out
#SBATCH --error=logs/compile_grad_probe_%A.err
set -euo pipefail
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"
unset TMPDIR
module load PrgEnv-cray
module load cuda/12.6
module load gcc-native/12.3
source .venv/bin/activate
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29511
srun --export=ALL .venv/bin/python scripts/probe/compile_grad_probe.py
