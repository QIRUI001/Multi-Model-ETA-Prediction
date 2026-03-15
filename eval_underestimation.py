"""Evaluate underestimation rates and per-duration-bin metrics for all models."""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Import models from baselines
from baselines import (LSTMModel, GRUModel, MLPModel, 
                       inverse_normalize_target, load_data)

def compute_detailed_metrics(y_pred, y_true):
    """Compute standard + operational metrics."""
    error = y_pred - y_true  # positive = overestimate, negative = underestimate
    abs_error = np.abs(error)
    
    mae = abs_error.mean()
    rmse = np.sqrt((error**2).mean())
    
    mask = y_true > 24
    mape = np.mean(np.abs(error[mask] / y_true[mask])) * 100 if mask.sum() > 0 else 0
    
    # Underestimation metrics
    under_mask = error < 0  # pred < true
    underestimation_rate = under_mask.mean() * 100
    mean_underestimation = np.abs(error[under_mask]).mean() if under_mask.sum() > 0 else 0
    
    # Severe underestimation (>24h late)
    severe_under = (error < -24).mean() * 100
    
    # Per-duration-bin MAE
    bins = [(0, 100), (100, 200), (200, 400), (400, 720)]
    bin_maes = {}
    for lo, hi in bins:
        m = (y_true >= lo) & (y_true < hi)
        if m.sum() > 0:
            bin_maes[f"{lo}-{hi}h"] = {
                'MAE': float(abs_error[m].mean()),
                'count': int(m.sum()),
                'underest_rate': float((error[m] < 0).mean() * 100)
            }
    
    return {
        'MAE_hours': float(mae),
        'RMSE': float(rmse),
        'MAPE': float(mape),
        'underestimation_rate': float(underestimation_rate),
        'mean_underestimation_hours': float(mean_underestimation),
        'severe_underestimation_rate_24h': float(severe_under),
        'per_bin': bin_maes
    }


def predict_model(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            out = model(x).squeeze(-1)
            preds.append(out.cpu().numpy())
    return np.concatenate(preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dir', type=str, required=True)
    parser.add_argument('--norm_path', type=str, default='./output/norm_params.npz')
    parser.add_argument('--baselines_dir', type=str, default='./output/baselines')
    parser.add_argument('--output_dir', type=str, default='./output/baselines')
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_train, y_train, X_val, y_val, X_test, y_test, sd_test, target_mean, target_std = \
        load_data(args.cache_dir, args.norm_path, args.batch_size, args.num_workers)
    
    y_true = inverse_normalize_target(np.array(y_test), target_mean, target_std)
    
    n_features = X_test.shape[2]
    seq_len = X_test.shape[1]
    
    test_ds = TensorDataset(torch.FloatTensor(np.array(X_test)), torch.FloatTensor(np.array(y_test)))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    
    results = {}
    
    # Evaluate saved baseline models
    for name, ModelClass, kwargs in [
        ('LSTM', LSTMModel, {'input_dim': n_features, 'hidden_dim': 256, 'num_layers': 2, 'dropout': 0.1}),
        ('GRU', GRUModel, {'input_dim': n_features, 'hidden_dim': 256, 'num_layers': 2, 'dropout': 0.1}),
        ('MLP', MLPModel, {'input_dim': n_features, 'seq_len': seq_len}),
    ]:
        model_path = os.path.join(args.baselines_dir, f'best_{name.lower()}.pth')
        if os.path.exists(model_path):
            print(f"\nEvaluating {name}...")
            model = ModelClass(**kwargs)
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model = model.to(device)
            y_pred_norm = predict_model(model, test_loader, device)
            y_pred = inverse_normalize_target(y_pred_norm, target_mean, target_std)
            y_pred = np.maximum(y_pred, 0)
            results[name] = compute_detailed_metrics(y_pred, y_true)
            print(f"  {name}: MAE={results[name]['MAE_hours']:.2f}h, "
                  f"Underest={results[name]['underestimation_rate']:.1f}%, "
                  f"Severe(<-24h)={results[name]['severe_underestimation_rate_24h']:.1f}%")
        else:
            print(f"  Skipping {name} (no saved model at {model_path})")
    
    # Evaluate Informer if available
    informer_path = os.path.join(os.path.dirname(args.baselines_dir), 'best_informer.pth')
    if os.path.exists(informer_path):
        print(f"\nEvaluating Informer...")
        # Need to import and set up Informer model
        from train_eta import MemmapDataset
        from src.informer.model import Informer
        
        # Load normalization from the dataset
        norm = np.load(args.norm_path)
        target_mean_f = float(norm['target_mean'])
        target_std_f = float(norm['target_std'])
        
        # Create Informer model with default config
        model = Informer(
            enc_in=n_features, dec_in=n_features, c_out=1,
            seq_len=seq_len, label_len=24, out_len=1,
            d_model=512, n_heads=8, e_layers=2, d_layers=1,
            d_ff=2048, dropout=0.05, attn='prob', embed='timeF',
            activation='gelu', output_attention=False, distil=True,
            device=device
        ).to(device)
        
        state = torch.load(informer_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        
        # Need full dataset with marks for Informer
        cache_dir = args.cache_dir
        X_test_mm = np.memmap(os.path.join(cache_dir, 'X_test.npy'), dtype='float32', mode='r').reshape(-1, seq_len, n_features)
        y_test_mm = np.memmap(os.path.join(cache_dir, 'y_test.npy'), dtype='float32', mode='r')
        
        # Load test marks
        mark_enc_path = os.path.join(cache_dir, 'X_mark_enc_test.npy')
        mark_dec_path = os.path.join(cache_dir, 'X_mark_dec_test.npy')
        
        if os.path.exists(mark_enc_path):
            X_mark_enc_test = np.memmap(mark_enc_path, dtype='float32', mode='r')
            n_test = len(y_test_mm)
            mark_dim = X_mark_enc_test.shape[0] // (n_test * seq_len)
            X_mark_enc_test = X_mark_enc_test.reshape(n_test, seq_len, mark_dim)
            
            label_len = 24
            dec_len = label_len + 1
            X_mark_dec_test = np.memmap(mark_dec_path, dtype='float32', mode='r').reshape(n_test, dec_len, mark_dim)
            
            # Build decoder input
            X_dec_test = np.zeros((n_test, dec_len, n_features), dtype='float32')
            X_dec_test[:, :label_len, :] = X_test_mm[:, -label_len:, :]
            
            # Predict in batches
            model.eval()
            preds = []
            bs = args.batch_size
            with torch.no_grad():
                for i in range(0, n_test, bs):
                    end = min(i + bs, n_test)
                    x_enc = torch.FloatTensor(np.array(X_test_mm[i:end])).to(device)
                    x_mark_enc = torch.FloatTensor(np.array(X_mark_enc_test[i:end])).to(device)
                    x_dec = torch.FloatTensor(np.array(X_dec_test[i:end])).to(device)
                    x_mark_dec = torch.FloatTensor(np.array(X_mark_dec_test[i:end])).to(device)
                    
                    out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                    preds.append(out.squeeze(-1).squeeze(-1).cpu().numpy())
            
            y_pred_norm = np.concatenate(preds)
            y_pred = np.expm1(y_pred_norm * target_std_f + target_mean_f)
            y_pred = np.maximum(y_pred, 0)
            y_true_inf = np.expm1(np.array(y_test_mm) * target_std_f + target_mean_f)
            
            results['Informer'] = compute_detailed_metrics(y_pred, y_true_inf)
            print(f"  Informer: MAE={results['Informer']['MAE_hours']:.2f}h, "
                  f"Underest={results['Informer']['underestimation_rate']:.1f}%, "
                  f"Severe(<-24h)={results['Informer']['severe_underestimation_rate_24h']:.1f}%")
        else:
            print("  Skipping Informer (no mark data found)")
    
    # Save results
    with open(os.path.join(args.output_dir, 'underestimation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary table
    print("\n" + "="*80)
    print("DETAILED COMPARISON")
    print("="*80)
    print(f"{'Model':<15} {'MAE(h)':<10} {'RMSE(h)':<10} {'MAPE(%)':<10} {'Underest%':<12} {'Severe%':<10}")
    print("-"*67)
    for name, m in results.items():
        print(f"{name:<15} {m['MAE_hours']:<10.2f} {m['RMSE']:<10.2f} {m['MAPE']:<10.2f} "
              f"{m['underestimation_rate']:<12.1f} {m['severe_underestimation_rate_24h']:<10.1f}")
    
    # Per-bin results
    print("\nPer-Duration-Bin MAE (hours):")
    bins = ['0-100h', '100-200h', '200-400h', '400-720h']
    header = f"{'Model':<15} " + " ".join(f"{b:<15}" for b in bins)
    print(header)
    print("-"*len(header))
    for name, m in results.items():
        row = f"{name:<15} "
        for b in bins:
            if b in m['per_bin']:
                row += f"{m['per_bin'][b]['MAE']:<15.2f}"
            else:
                row += f"{'N/A':<15}"
        print(row)


if __name__ == '__main__':
    main()
