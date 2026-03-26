#!/bin/bash
# Self-distillation and checkpoint averaging experiments
# Phase 1: Generate soft targets from single best model (no ensemble needed)
# Phase 2: Train students with self-distillation
# Phase 3: KD + checkpoint averaging

set -e
cd /root/autodl-tmp/Multi-Model-ETA-Prediction
PY=/root/miniconda3/bin/python
CACHE=output/cache_sequences/seq48_label24_pred1_mv150000_ms50000000

echo "============================================"
echo "Phase 1: Generate self-distillation targets"
echo "============================================"

# Generate soft targets from the best single model (15.28h)
$PY generate_soft_targets.py \
    --teacher_ckpt output/mstgn_mlp2_mse/best_mstgn.pth \
    --output_subdir soft_targets_self

echo ""
echo "============================================"
echo "Phase 2: Self-distillation experiments"
echo "============================================"

# Self-distillation α=0.5
echo "--- Self-distill α=0.5 ---"
$PY train_mstgn.py \
    --variant mlp2 --loss mse --scheduler plateau \
    --output_dir output/mstgn_selfdistill_a05 \
    --distill --distill_alpha 0.5 \
    --soft_targets_dir $CACHE/soft_targets_self \
    --seed 42 --epochs 15

# Self-distillation α=0.6
echo "--- Self-distill α=0.6 ---"
$PY train_mstgn.py \
    --variant mlp2 --loss mse --scheduler plateau \
    --output_dir output/mstgn_selfdistill_a06 \
    --distill --distill_alpha 0.6 \
    --soft_targets_dir $CACHE/soft_targets_self \
    --seed 42 --epochs 15

# Self-distillation α=0.4
echo "--- Self-distill α=0.4 ---"
$PY train_mstgn.py \
    --variant mlp2 --loss mse --scheduler plateau \
    --output_dir output/mstgn_selfdistill_a04 \
    --distill --distill_alpha 0.4 \
    --soft_targets_dir $CACHE/soft_targets_self \
    --seed 42 --epochs 15

echo ""
echo "============================================"
echo "Phase 3: Ensemble KD + checkpoint averaging"
echo "============================================"

# KD α=0.6 + top-3 checkpoint averaging
echo "--- KD α=0.6 + ckpt_avg=3 ---"
$PY train_mstgn.py \
    --variant mlp2 --loss mse --scheduler plateau \
    --output_dir output/mstgn_kd06_avg3 \
    --distill --distill_alpha 0.6 \
    --ckpt_avg 3 \
    --seed 42 --epochs 15

# KD α=0.5 + top-3 checkpoint averaging
echo "--- KD α=0.5 + ckpt_avg=3 ---"
$PY train_mstgn.py \
    --variant mlp2 --loss mse --scheduler plateau \
    --output_dir output/mstgn_kd05_avg3 \
    --distill --distill_alpha 0.5 \
    --ckpt_avg 3 \
    --seed 42 --epochs 15

# KD α=0.6 + top-5 checkpoint averaging (more checkpoints)
echo "--- KD α=0.6 + ckpt_avg=5 ---"
$PY train_mstgn.py \
    --variant mlp2 --loss mse --scheduler plateau \
    --output_dir output/mstgn_kd06_avg5 \
    --distill --distill_alpha 0.6 \
    --ckpt_avg 5 \
    --seed 42 --epochs 15

echo ""
echo "============================================"
echo "All experiments complete!"
echo "============================================"
