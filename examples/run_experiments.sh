#!/bin/bash
# Run all data filtering experiments for the results table.
# Usage: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 bash examples/run_experiments.sh
set -e

NUM_GPUS="${NUM_GPUS:-7}"
NUM_EXAMPLES=250  # ~5% of 4750 train examples
MAX_SAMPLES=5000
NUM_EPOCHS=3
SEED=42

echo "=== Running all filter experiments (n=$NUM_EXAMPLES, ~5% of data) ==="

# Step 1: LoRA SFT on full dataset (needed for trackstar-lora)
echo "--- Step 1: LoRA SFT on full dataset ---"
CUDA_VISIBLE_DEVICES=0 python -m examples.filter_data \
  --filter trackstar --use_lora --num_examples $NUM_EXAMPLES \
  --max_samples $MAX_SAMPLES --num_epochs $NUM_EPOCHS --seed $SEED --dry_run

# Step 2: Trackstar-LoRA pipeline (scoring)
echo "--- Step 2: Trackstar-LoRA scoring pipeline ---"
CUDA_VISIBLE_DEVICES=0 python -m examples.filter_data \
  --filter trackstar --use_lora --num_examples $NUM_EXAMPLES \
  --max_samples $MAX_SAMPLES --num_epochs $NUM_EPOCHS --seed $SEED

# Step 3: Random baseline at 5%
echo "--- Step 3: Random at 5% ---"
CUDA_VISIBLE_DEVICES=0 python -m examples.filter_data \
  --filter random --num_examples $NUM_EXAMPLES \
  --max_samples $MAX_SAMPLES --num_epochs $NUM_EPOCHS --seed $SEED

# Step 4: Attribution at 5%
echo "--- Step 4: Attribution at 5% ---"
CUDA_VISIBLE_DEVICES=0 python -m examples.filter_data \
  --filter attribution --num_examples $NUM_EXAMPLES \
  --max_samples $MAX_SAMPLES --num_epochs $NUM_EPOCHS --seed $SEED

# Step 5: Trackstar (full SFT) at 5%
echo "--- Step 5: Trackstar at 5% ---"
CUDA_VISIBLE_DEVICES=0 python -m examples.filter_data \
  --filter trackstar --num_examples $NUM_EXAMPLES \
  --max_samples $MAX_SAMPLES --num_epochs $NUM_EPOCHS --seed $SEED

echo "=== All experiments complete ==="
