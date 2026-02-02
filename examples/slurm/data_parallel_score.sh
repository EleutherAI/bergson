#!/usr/bin/bash
#SBATCH --job-name=bergson_score_data_parallel
#SBATCH --nodes=64
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/bergson_score_data_parallel_%A_%N.out
#SBATCH --error=logs/bergson_score_data_parallel_%A_%N.err

mkdir -p logs

hf auth login --token <HUGGINGFACE_TOKEN>

# Set number of nodes
NUM_NODES=64
RUN_NAME="bergson_score"

TOTAL_EXAMPLES=$(cat dataset_size.txt)
EXAMPLES_PER_NODE=$((TOTAL_EXAMPLES / NUM_NODES))

# Export variables for the worker script
export TOTAL_EXAMPLES
export EXAMPLES_PER_NODE
export NUM_NODES
export RUN_NAME

# Run worker script on each node
srun --kill-on-bad-exit=1 --output=logs/bergson_score_%A_%t.out --error=logs/bergson_score_%A_%t.err bash score_worker.sh
