#!/bin/bash

# Run EKFAC application tests

cd "$(dirname "$0")"

pytest -s -v \
    --test_dir "./test_files/pile_10k_examples" \
    --gradient_path "./test_files/pile_10k_examples/test_gradients/proj_dim_0" \
    --overwrite \
    --use_fsdp \
    --world_size 8 \
    --gradient_batch_size 10 \
    test_apply_ekfac.py
