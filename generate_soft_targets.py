#!/usr/bin/env python
"""
Generate soft targets from ensemble for knowledge distillation.

Loads the top-K seed models, averages their predictions on train/val/test,
and saves as .npy files alongside the data cache.

Usage:
    python generate_soft_targets.py
    python generate_soft_targets.py --top_k 7 --ensemble_dir output/mstgn_ensemble
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from src.mstgn.model import MSTGN_MLP2


class MSTGNDataset(Dataset):
    def __init__(self, X_path, cell_ids_path, y_path, actual_length=None):
        self.X = np.load(X_path, mmap_mode='r')
        self.cell_ids = np.load(cell_ids_path, mmap_mode='r')
        self.y = np.load(y_path, mmap_mode='r')
        self.length = actual_length if actual_length is not None else len(self.y)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx].copy()).float(),
            torch.from_numpy(self.cell_ids[idx].copy()).long(),
            torch.tensor(self.y[idx]).float()
        )


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    preds = []
    for x, cell_ids, _ in tqdm(loader, leave=False):
        x, cell_ids = x.to(device), cell_ids.to(device)
        pred = model(x, cell_ids)
        preds.append(pred.cpu().numpy())
    return np.concatenate(preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dir', default='output/cache_sequences/seq48_label24_pred1_mv150000_ms50000000')
    parser.add_argument('--graph_dir', default='output/graph')
    parser.add_argument('--ensemble_dir', default='output/mstgn_ensemble')
    parser.add_argument('--top_k', type=int, default=7)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache_dir = Path(args.cache_dir)
    graph_dir = Path(args.graph_dir)
    ensemble_dir = Path(args.ensemble_dir)

    # Load graph
    adj = np.load(graph_dir / 'adj_normalized.npy')
    node_features = np.load(graph_dir / 'node_features.npy')

    # Find seed dirs and rank by val loss
    seed_dirs = sorted(ensemble_dir.glob('seed_*'))
    seed_results = []
    for sd in seed_dirs:
        rfile = sd / 'results.json'
        if rfile.exists():
            with open(rfile) as f:
                r = json.load(f)
            seed_results.append((sd, r.get('best_val_loss', float('inf'))))

    seed_results.sort(key=lambda x: x[1])
    top_dirs = [sd for sd, _ in seed_results[:args.top_k]]
    print(f"Using top-{args.top_k} models:")
    for sd, vl in seed_results[:args.top_k]:
        print(f"  {sd.name}: val_loss={vl:.6f}")

    # Load data
    counts = np.load(cache_dir / 'actual_counts.npy', allow_pickle=True).item()
    splits = {}
    for split in ['train', 'val', 'test']:
        ds = MSTGNDataset(
            cache_dir / f'X_{split}.npy',
            cache_dir / f'cell_ids_{split}.npy',
            cache_dir / f'y_{split}.npy',
            counts[split]
        )
        loader = DataLoader(ds, batch_size=args.batch_size,
                           num_workers=args.num_workers, pin_memory=True)
        splits[split] = (ds, loader)
        print(f"  {split}: {counts[split]:,} samples")

    # Generate predictions from each model
    all_preds = {s: [] for s in splits}

    for sd in top_dirs:
        print(f"\nLoading {sd.name}...")
        model = MSTGN_MLP2(
            adj_matrix=adj,
            init_node_features=node_features,
            seq_feat_dim=11, seq_len=48,
            gcn_hidden=64, cell_emb_dim=32,
            dropout=0.1
        ).to(device)
        state = torch.load(sd / 'best_mstgn.pth', map_location=device, weights_only=True)
        model.load_state_dict(state)

        for split, (ds, loader) in splits.items():
            print(f"  Predicting {split}...")
            preds = predict_all(model, loader, device)
            all_preds[split].append(preds)

        del model
        torch.cuda.empty_cache()

    # Average and save
    output_dir = cache_dir / 'soft_targets'
    output_dir.mkdir(exist_ok=True)

    for split in splits:
        stacked = np.stack(all_preds[split], axis=0)
        soft = stacked.mean(axis=0).astype(np.float32)
        outpath = output_dir / f'y_soft_{split}.npy'
        np.save(outpath, soft)
        print(f"Saved {outpath}: shape={soft.shape}, mean={soft.mean():.4f}, std={soft.std():.4f}")

    print(f"\nSoft targets saved to {output_dir}")
    print("Done!")


if __name__ == '__main__':
    main()
