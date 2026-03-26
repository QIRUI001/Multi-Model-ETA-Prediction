#!/bin/bash
# Train MSTGN-MLP2+MSE models with different seeds, then evaluate ensemble.
set -e

PYTHON=/root/miniconda3/bin/python
SEEDS=(42 43 44 45 46 47 48 49 50 51 52 53 54 55 56)

echo "============================================"
echo "Training ${#SEEDS[@]}-model ensemble (MLP2 + MSE)"
echo "============================================"

for seed in "${SEEDS[@]}"; do
    OUT_DIR="output/ensemble/seed${seed}"
    if [ -f "${OUT_DIR}/predictions.npz" ]; then
        echo ">>> Seed $seed already trained, skipping."
        continue
    fi
    echo ""
    echo ">>> Training seed=$seed ..."
    $PYTHON train_mstgn.py \
        --variant mlp2 \
        --loss mse \
        --epochs 15 \
        --lr 1e-3 \
        --patience 4 \
        --seed $seed \
        --output_dir $OUT_DIR
done

echo ""
echo ">>> All models trained. Evaluating ensemble..."
$PYTHON eval_ensemble.py

echo ""
echo "============================================"
echo "Ensemble training and evaluation complete!"
echo "============================================"
