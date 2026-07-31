#!/usr/bin/env bash
# Wait for the docnorm SOURCE run to finish, then start the toknorm one.
# Both need all 8 GPUs, so they cannot overlap.
#
# Bounded: gives up at DEADLINE_S, and exits early if the docnorm process
# disappears without producing scores.
set -u

REPO=/mnt/ssd-1/lucia/bergson-damping
cd "$REPO" || exit 1

DEADLINE_S=$((6 * 3600))
DOC_SCORES=runs/lotus_source_q50_docnorm/scores.npy
DOC_VALIDATE=runs/lotus_source_q50_docnorm_validate/summary.csv
TOK_CFG=examples/method_comparison/gpt2_wikitext_lotus_source_q50_toknorm.yaml

start=$(date +%s)
while true; do
    now=$(date +%s)
    elapsed=$((now - start))
    if [ "$elapsed" -gt "$DEADLINE_S" ]; then
        echo "CHAIN: deadline of ${DEADLINE_S}s reached; not launching toknorm"
        exit 1
    fi

    if [ -f "$DOC_VALIDATE" ]; then
        echo "CHAIN: docnorm validation complete after ${elapsed}s"
        break
    fi

    if ! pgrep -f "bergson.*docnorm" >/dev/null 2>&1; then
        # Process gone. Only proceed if it actually got to the end.
        if [ -f "$DOC_VALIDATE" ]; then
            echo "CHAIN: docnorm finished after ${elapsed}s"
            break
        fi
        echo "CHAIN: docnorm process gone after ${elapsed}s without $DOC_VALIDATE" \
             "(scores.npy exists: $([ -f "$DOC_SCORES" ] && echo yes || echo no)); aborting"
        exit 1
    fi

    sleep 60
done

CMD="PYTHONPATH=$REPO python -m bergson $TOK_CFG"
echo "CHAIN: launching toknorm"
echo "CMD: $CMD"
echo "CMD: $CMD" > logs/source_toknorm.log
PYTHONPATH="$REPO" python -m bergson "$TOK_CFG" >> logs/source_toknorm.log 2>&1
echo "CHAIN: toknorm exited with $?"
