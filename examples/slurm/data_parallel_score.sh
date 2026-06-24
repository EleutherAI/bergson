#!/usr/bin/bash
#SBATCH --job-name=bergson_score_data_parallel
#SBATCH --array=0-63
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/bergson_score_%A_%a.out
#SBATCH --error=logs/bergson_score_%A_%a.err

# Embarrassingly parallel scoring: every array task processes one contiguous
# shard of the dataset and publishes it into the same run_path. A task that
# dies is requeued by SLURM and rebuilds only its own shard; shards that
# already finished are skipped, so requeues are idempotent. No stitching is
# needed afterwards — bergson reads run_path as one index.
#
# Check progress at any time with: bergson status runs/$RUN_NAME

mkdir -p logs

hf auth login --token <HUGGINGFACE_TOKEN>

NUM_SHARDS=64  # keep equal to the array size above
RUN_NAME="bergson_score"

# --shard_id is inferred from SLURM_ARRAY_TASK_ID
python -m bergson score \
    runs/$RUN_NAME \
    --num_shards $NUM_SHARDS \
    --query_path runs/$QUERY_RUN_NAME \
    --score individual \
    --dataset NeelNanda/pile-10k \
    --model EleutherAI/pythia-14m \
    --token_batch_size 15500 \
    --truncation
