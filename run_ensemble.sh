#!/bin/bash
# Train 5 MSTGN-MLP2+MSE models with different seeds, then evaluate ensemble.
set -e

PYTHON=/root/miniconda3/bin/python
SEEDS=(42 43 44 45 46)

echo "============================================"
echo "Training 5-model ensemble (MLP2 + MSE)"
echo "============================================"

for seed in "${SEEDS[@]}"; do
    echo ""
    echo ">>> Training seed=$seed ..."
    $PYTHON train_mstgn.py \
        --variant mlp2 \
        --loss mse \
        --epochs 15 \
        --lr 1e-3 \
        --patience 4 \
        --seed $seed \
        --output_dir output/ensemble/seed${seed}
done

echo ""
echo ">>> All 5 models trained. Evaluating ensemble..."
$PYTHON eval_ensemble.py

echo ""
echo "============================================"
echo "Ensemble training and evaluation complete!"
echo "============================================"
