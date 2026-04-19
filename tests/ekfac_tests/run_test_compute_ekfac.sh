#!/bin/bash

# Run EKFAC computation tests

cd "$(dirname "$0")"

pytest -s -v \
    --test_dir "./test_files/pile_10k_examples" \
    --world_size 8 \
    --n_samples 10000 \
    --use_fsdp \
    --overwrite \
    test_compute_ekfac.py \
    test_covariance.py \
    test_ekfac_recomposition.py
