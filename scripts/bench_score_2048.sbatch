#!/bin/bash
#SBATCH --job-name=bench-s2048
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:30:00
#SBATCH --output=/lus/lfs1aip2/projects/public/a6a/lucia/bergson3/runs/bench-s2048-%j.out

module load PrgEnv-cray
module load cuda/12.6

export HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_HOME=/home/a6a/lucia.a6a/.cache/torch
export TORCHINDUCTOR_CACHE_DIR=/home/a6a/lucia.a6a/.cache/torch_inductor

cd /lus/lfs1aip2/projects/public/a6a/lucia/bergson3
source .venv/bin/activate

echo "Scoring with token_batch_size=2048"
time bergson trackstar runs/bench_score_bf16_2048     --model allenai/Olmo-3-7B-Instruct     --normalizer adafactor     --stats_sample_size 10000     --index_cfg.precision bf16     --data.dataset EleutherAI/SmolLM2-135M-10B     --data.split "train[:10000]"     --data.truncation     --query.dataset cais/mmlu     --query.subset sociology     --query.split test     --query.format_template bergson/templates/mcqa.yaml     --query.truncation     --unit_normalize     --aggregation mean     --normalize_aggregated_grad     --projection_dim 16     --token_batch_size 2048     --nproc_per_node 4     --resume
