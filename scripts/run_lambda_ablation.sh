#!/usr/bin/env bash
# Sequential KL-disagreement λ ablation.
# Matches the existing λ=0.3 config: kl_top_k=3, kl_every_n=8, bs=4, grad_accum=8, lr=1e-5, epochs=10.
set -euo pipefail

cd "$(dirname "$0")/.."

LOGDIR=logs
mkdir -p "$LOGDIR" models

run_lambda() {
    local LAM=$1
    local TAG="lambda${LAM}"
    local OUT="models/joint-kl-${TAG}"
    local LOG="${LOGDIR}/kl_${TAG}.log"

    if [ -f "${OUT}/training_log.json" ]; then
        echo "[skip] ${OUT}/training_log.json already exists"
        return 0
    fi

    echo "=== Starting λ=${LAM} at $(date -Is) ==="
    python scripts/12_finetune_kl_disagreement.py \
        --lambda-kl "${LAM}" \
        --kl-top-k 3 \
        --kl-every-n 8 \
        --epochs 10 \
        --batch-size 4 \
        --grad-accum 8 \
        --lr 1e-5 \
        --neg-ratio 3.0 \
        --max-length 256 \
        --patience 3 \
        --seed 42 \
        --output-dir "${OUT}" \
        >> "${LOG}" 2>&1
    echo "=== Finished λ=${LAM} at $(date -Is) ==="
}

for LAM in 0.0 0.1 0.5; do
    run_lambda "${LAM}"
done

echo "All ablation runs completed at $(date -Is)"
