#!/usr/bin/env bash
# Run the modula vs baseline MAGIC attribution experiment.
#
# Trains GPT-2 on WikiText-2 with and without modular norm normalization,
# then compares the Spearman rho of the attribution scores.
#
# Prerequisites:
#   pip install modula
#
# Usage:
#   bash examples/modula/run_experiment.sh
#
# Adjust nproc_per_node and batch_size in the YAML files for your GPU count.
# The per-GPU batch must be 16 to avoid OOM in the backward pass.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

echo "=== Running baseline (no modular norm) ==="
echo "bergson examples/modula/baseline.yaml"
bergson examples/modula/baseline.yaml

echo ""
echo "=== Running modula (with modular norm) ==="
echo "bergson examples/modula/modula.yaml"
bergson examples/modula/modula.yaml

echo ""
echo "=== Results ==="
echo "Baseline:"
cat runs/modula_experiment/baseline/summary.csv
echo ""
echo "Modula:"
cat runs/modula_experiment/modula/summary.csv
