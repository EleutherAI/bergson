#!/bin/bash

# Run EKFAC computation tests

cd "$(dirname "$0")"

pytest -s -v \
    --test_dir "./test_files/pile_100_examples" \
    --world_size 8 \
    --overwrite \
    test_compute_ekfac.py \
    test_covariance.py \
    test_eigenvectors.py \
    test_eigenvalue_correction.py
