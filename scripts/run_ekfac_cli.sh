#!/usr/bin/env bash
set -euo pipefail

# ── User Configuration ───────────────────────────────────────────────────────
MODEL=""
QUERY_DATASET=""
QUERY_SPLIT=""
INDEX_DATASET=""
INDEX_SPLIT=""
RUN_PATH=""

# ── Data ─────────────────────────────────────────────────────────────────────
PROMPT_COLUMN=""
COMPLETION_COLUMN=""
CONVERSATION_COLUMN=""
TRUNCATION="--data.truncation --query.truncation"

# ── HessianConfig ────────────────────────────────────────────────────────────
HESSIAN_METHOD="kfac"
HESSIAN_DTYPE="bf16"


# ── IndexConfig ──────────────────────────────────────────────────────────────
PRECISION="bf16"
TOKEN_BATCH_SIZE=2048
LOSS_FN="ce"
LOSS_REDUCTION="sum"
FSDP="--fsdp"

# Module filtering – glob pattern to EXCLUDE modules from collection.
FILTER_MODULES=""

# ── EKFAC ────────────────────────────────────────────────────────────────────
LAMBDA_DAMP_FACTOR=0.1


# ── Build & run command ──────────────────────────────────────────────────────
CMD="bergson ekfac ${RUN_PATH}"
CMD+=" --query.dataset ${QUERY_DATASET}"
CMD+=" --query.split ${QUERY_SPLIT}"
CMD+=" --data.dataset ${INDEX_DATASET}"
CMD+=" --data.split ${INDEX_SPLIT}"
CMD+=" --model ${MODEL}"
CMD+=" --data.prompt_column ${PROMPT_COLUMN}"
CMD+=" --query.prompt_column ${PROMPT_COLUMN}"
CMD+=" --method ${HESSIAN_METHOD}"
CMD+=" --hessian_dtype ${HESSIAN_DTYPE}"
CMD+=" --index_cfg.precision ${PRECISION}"
CMD+=" --token_batch_size ${TOKEN_BATCH_SIZE}"
CMD+=" --loss_fn ${LOSS_FN}"
CMD+=" --loss_reduction ${LOSS_REDUCTION}"
CMD+=" --lambda_damp_factor ${LAMBDA_DAMP_FACTOR}"
CMD+=" --overwrite"
CMD+=" --skip_preconditioners"

# Optional flags – appended only when non-empty
[[ -n "${COMPLETION_COLUMN}" ]]     && CMD+=" --data.completion_column ${COMPLETION_COLUMN}"
[[ -n "${CONVERSATION_COLUMN}" ]]   && CMD+=" --data.conversation_column ${CONVERSATION_COLUMN}"
[[ -n "${FILTER_MODULES}" ]]        && CMD+=" --filter_modules \"${FILTER_MODULES}\""
[[ -n "${TRUNCATION}" ]]            && CMD+=" ${TRUNCATION}"
[[ -n "${FSDP}" ]]                  && CMD+=" ${FSDP}"

echo "Running: ${CMD}"
eval " ${CMD}"
