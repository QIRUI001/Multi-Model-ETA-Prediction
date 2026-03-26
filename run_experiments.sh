#!/bin/bash
# Batch experiments: FT-Transformer tuning + Knowledge Distillation
# Run all experiments sequentially, shutdown after last one
set -e

PYTHON=/root/miniconda3/bin/python
cd /root/autodl-tmp/Multi-Model-ETA-Prediction

echo "=========================================="
echo "Step 0: Generate soft targets from ensemble"
echo "=========================================="
if [ ! -f "output/cache_sequences/seq48_label24_pred1_mv150000_ms50000000/soft_targets/y_soft_train.npy" ]; then
    $PYTHON generate_soft_targets.py --top_k 7
else
    echo "Soft targets already exist, skipping."
fi

echo ""
echo "=========================================="
echo "Exp 1: FT-Transformer lr=3e-4, cosine"
echo "=========================================="
$PYTHON train_mstgn.py \
    --variant ft_transformer --loss mse \
    --lr 3e-4 --scheduler cosine --epochs 30 --dropout 0.15 \
    --output_dir output/mstgn_ftt_lr3e4_cos --seed 42

echo ""
echo "=========================================="
echo "Exp 2: FT-Transformer lr=1e-4, cosine"
echo "=========================================="
$PYTHON train_mstgn.py \
    --variant ft_transformer --loss mse \
    --lr 1e-4 --scheduler cosine --epochs 30 --dropout 0.15 \
    --output_dir output/mstgn_ftt_lr1e4_cos --seed 42

echo ""
echo "=========================================="
echo "Exp 3: MLP2 + Knowledge Distillation (alpha=0.5)"
echo "=========================================="
$PYTHON train_mstgn.py \
    --variant mlp2 --loss mse \
    --distill --distill_alpha 0.5 \
    --output_dir output/mstgn_mlp2_kd05 --seed 42

echo ""
echo "=========================================="
echo "Exp 4: MLP2 + Knowledge Distillation (alpha=0.3)"
echo "=========================================="
$PYTHON train_mstgn.py \
    --variant mlp2 --loss mse \
    --distill --distill_alpha 0.3 \
    --output_dir output/mstgn_mlp2_kd03 --seed 42

echo ""
echo "=========================================="
echo "Exp 5: MLP2 + KD (alpha=0.5, seed=52)"
echo "=========================================="
$PYTHON train_mstgn.py \
    --variant mlp2 --loss mse \
    --distill --distill_alpha 0.5 \
    --output_dir output/mstgn_mlp2_kd05_s52 --seed 52

echo ""
echo "=========================================="
echo "Exp 6: FT-Transformer + KD (alpha=0.5)"
echo "=========================================="
$PYTHON train_mstgn.py \
    --variant ft_transformer --loss mse \
    --lr 3e-4 --scheduler cosine --epochs 30 --dropout 0.15 \
    --distill --distill_alpha 0.5 \
    --output_dir output/mstgn_ftt_kd05 --seed 42

echo ""
echo "=========================================="
echo "All experiments complete! Collecting results..."
echo "=========================================="

for dir in output/mstgn_ftt_lr3e4_cos output/mstgn_ftt_lr1e4_cos output/mstgn_mlp2_kd05 output/mstgn_mlp2_kd03 output/mstgn_mlp2_kd05_s52 output/mstgn_ftt_kd05; do
    if [ -f "$dir/results.json" ]; then
        mae=$(python3 -c "import json; r=json.load(open('$dir/results.json')); print(f\"{r['model']}: MAE={r['metrics']['MAE_hours']:.2f}h\")")
        echo "$mae  ($dir)"
    fi
done

echo ""
echo "Done! Shutting down..."
