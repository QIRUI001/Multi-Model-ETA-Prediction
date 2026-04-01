#!/bin/bash
# Reviewer Ablation Experiments
# Q1: Grid resolution ablation (1°, 2°, 4°)
# Q3: XGBoost + GCN embeddings ablation
set -e

PYTHON=/root/miniconda3/bin/python
CACHE_DIR="output/cache_sequences/seq48_label24_pred1_mv150000_ms50000000"
cd /root/autodl-tmp/Multi-Model-ETA-Prediction

echo "=========================================="
echo "Q1a: Build 1°×1° route graph"
echo "=========================================="
$PYTHON build_route_graph.py \
    --cell_size 1.0 \
    --output_dir output/graph_1deg \
    --cell_ids_dir output/graph_1deg

echo ""
echo "=========================================="
echo "Q1b: Build 4°×4° route graph"
echo "=========================================="
$PYTHON build_route_graph.py \
    --cell_size 4.0 \
    --output_dir output/graph_4deg \
    --cell_ids_dir output/graph_4deg

echo ""
echo "=========================================="
echo "Q1c: Rebuild 2°×2° graph (restore default cell_ids)"
echo "=========================================="
$PYTHON build_route_graph.py \
    --cell_size 2.0 \
    --output_dir output/graph

echo ""
echo "=========================================="
echo "Q1d: Train MSTGN-MLP2 with 1°×1° grid"
echo "=========================================="
$PYTHON train_mstgn.py \
    --variant mlp2 --loss mse \
    --graph_dir output/graph_1deg \
    --cell_ids_dir output/graph_1deg \
    --output_dir output/mstgn_mlp2_1deg \
    --seed 42

echo ""
echo "=========================================="
echo "Q1e: Train MSTGN-MLP2 with 4°×4° grid"
echo "=========================================="
$PYTHON train_mstgn.py \
    --variant mlp2 --loss mse \
    --graph_dir output/graph_4deg \
    --cell_ids_dir output/graph_4deg \
    --output_dir output/mstgn_mlp2_4deg \
    --seed 42

echo ""
echo "=========================================="
echo "Q3: XGBoost + GCN embeddings (from best MSTGN-MLP2-KD checkpoint)"
echo "=========================================="
$PYTHON run_gcn_xgboost.py \
    --ckpt output/mstgn_mlp2_kd06/best_mstgn.pth \
    --graph_dir output/graph \
    --output_dir output/xgb_gcn

echo ""
echo "=========================================="
echo "All reviewer ablations complete!"
echo "Results in:"
echo "  Q1 1deg: output/mstgn_mlp2_1deg/results.json"
echo "  Q1 4deg: output/mstgn_mlp2_4deg/results.json"
echo "  Q3:      output/xgb_gcn/results.json"
echo "=========================================="

# Send summary to Discord
$PYTHON - <<'PYEOF'
import json, urllib.request

def read_mae(path):
    try:
        with open(path) as f:
            r = json.load(f)
        return r['metrics']['MAE_hours']
    except Exception:
        return None

mae_1deg = read_mae('output/mstgn_mlp2_1deg/results.json')
mae_4deg = read_mae('output/mstgn_mlp2_4deg/results.json')

xgb_results = {}
try:
    with open('output/xgb_gcn/results.json') as f:
        xgb_results = json.load(f)
except Exception:
    pass

msg = (
    f"✅ Reviewer Ablations Complete!\n"
    f"Q1 Grid Resolution (MSTGN-MLP2 no KD):\n"
    f"  1°×1°: MAE={mae_1deg:.2f}h\n"
    f"  2°×2°: MAE=15.28h (reference)\n"
    f"  4°×4°: MAE={mae_4deg:.2f}h\n"
)
if xgb_results:
    m_stat = xgb_results.get('xgb_stats_only', {})
    m_gcn = xgb_results.get('xgb_stats_gcn', {})
    msg += (
        f"\nQ3 XGBoost Ablation:\n"
        f"  XGBoost (Stats-only 85d): MAE={m_stat.get('MAE_hours', '?'):.2f}h\n"
        f"  XGBoost (Stats+GCN 177d): MAE={m_gcn.get('MAE_hours', '?'):.2f}h\n"
        f"  MSTGN-MLP2+KD: MAE=15.13h (reference)"
    )

webhook = "https://discord.com/api/webhooks/1477180434474336266/HQSS_BKlo1Ib-rzwItazg8ay0To2l-GR1GprnTvlPVpLxCxD9aBd4iHNgX"
data = json.dumps({"content": msg}).encode('utf-8')
req = urllib.request.Request(webhook, data=data, headers={'Content-Type': 'application/json'})
try:
    urllib.request.urlopen(req)
except Exception as e:
    print(f"Discord notification failed: {e}")
print(msg)
PYEOF
