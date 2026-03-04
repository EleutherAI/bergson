#!/bin/bash
#SBATCH --job-name=magic-bs-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --time=00:30:00
#SBATCH --output=/home/a6a/lucia.a6a/bergson3/runs/magic-bs-test-%j.out

module load PrgEnv-cray cuda/12.6
source /home/a6a/lucia.a6a/bergson3/.venv/bin/activate
cd /home/a6a/lucia.a6a/bergson3

export PYTORCH_ALLOC_CONF=expandable_segments:True

CKPT_BASE=/projects/a6a/public/lucia
OUT_BASE=/home/a6a/lucia.a6a/bergson3/runs

# Test batch sizes from largest to smallest (all divide 10000 evenly)
for BS in 16 10 8; do
    echo ""
    echo "=========================================="
    echo "Testing batch_size=$BS ($(($BS / 4)) per GPU)"
    echo "=========================================="

    # Clean up any previous test data
    rm -rf ${CKPT_BASE}/magic_wmdp_forget_0k_msl1024_ckpts/
    rm -rf ${OUT_BASE}/magic_wmdp_forget_0k_wmdp_msl1024_output/

    # Run full pipeline on tiny dataset: train + eval + backward
    python -u runs/magic_per_token.py \
        --dataset wmdp_forget \
        --num_examples $BS \
        --batch_size $BS \
        --eval_task wmdp

    EXIT_CODE=$?
    echo "batch_size=$BS exit code: $EXIT_CODE"

    # Clean up test data
    rm -rf ${CKPT_BASE}/magic_wmdp_forget_0k_msl1024_ckpts/
    rm -rf ${OUT_BASE}/magic_wmdp_forget_0k_wmdp_msl1024_output/

    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "SUCCESS: batch_size=$BS works"
        echo "For 10k examples: $((10000 / $BS)) steps"
        exit 0
    else
        echo "FAILED: batch_size=$BS OOM"
    fi
done

echo ""
echo "All batch sizes failed"
exit 1
