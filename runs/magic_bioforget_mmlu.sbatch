#!/bin/bash
#SBATCH --job-name=magic-bf-mmlu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --output=/home/a6a/lucia.a6a/bergson3/runs/magic-bf-mmlu-%j.out

module load PrgEnv-cray
module load cuda/12.6

export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_HOME=/home/a6a/lucia.a6a/.cache/torch
export TORCHINDUCTOR_CACHE_DIR=/home/a6a/lucia.a6a/.cache/torch_inductor

cd /home/a6a/lucia.a6a/bergson3
source .venv/bin/activate

python -u -m bergson.magic_wmdp \
    --train_data bio_forget \
    --eval_task mmlu \
    --lr 5e-5 \
    --batch_size 4 \
    --num_batches 250 \
    --max_seq_len 1024 \
    --ckpt_dir /projects/a6a/public/lucia/magic_bioforget_msl1024_ckpts \
    --output_dir /home/a6a/lucia.a6a/bergson3/runs/magic_bioforget_mmlu_msl1024_output \
    --eval_chunk_size 4
