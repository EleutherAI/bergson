#!/bin/bash
#SBATCH --job-name=from_local_probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:15:00
#SBATCH --output=logs/from_local_probe_%A.out
#SBATCH --error=logs/from_local_probe_%A.err
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
srun --export=ALL .venv/bin/torchrun --nnodes=1 --nproc_per_node=4 \
    --master_addr="$MASTER_ADDR" --master_port=29512 \
    scripts/probe/from_local_probe.py
