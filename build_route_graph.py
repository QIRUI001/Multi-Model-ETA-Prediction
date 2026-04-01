#!/usr/bin/env python
"""
Build a maritime route graph from processed AIS data.

Creates a spatial graph where:
- Nodes = ocean grid cells with significant shipping activity
- Edges = vessel transition frequency + geographic adjacency
- Node features = aggregated statistics per cell

Also computes cell_id mappings for preprocessed sequence data.

Usage:
    python build_route_graph.py
    python build_route_graph.py --cell_size 1.0  # finer grid
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# === Grid Configuration ===
LAT_MIN, LAT_MAX = -58.0, 70.0
LON_MIN, LON_MAX = -180.0, 180.0
MIN_CELL_COUNT = 50


def get_grid_params(cell_size):
    num_lat = int((LAT_MAX - LAT_MIN) / cell_size)
    num_lon = int((LON_MAX - LON_MIN) / cell_size)
    return num_lat, num_lon


def latlon_to_raw_cell(lat, lon, cell_size, num_lat, num_lon):
    """Map lat/lon arrays to raw cell indices."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    lat_idx = np.clip(((lat - LAT_MIN) / cell_size).astype(np.int32), 0, num_lat - 1)
    lon_idx = np.clip(((lon - LON_MIN) / cell_size).astype(np.int32), 0, num_lon - 1)
    return lat_idx * num_lon + lon_idx


def build_graph(args):
    cell_size = args.cell_size
    num_lat, num_lon = get_grid_params(cell_size)
    max_raw_id = num_lat * num_lon
    print(f"Grid: {num_lat}×{num_lon} = {max_raw_id} cells at {cell_size}° resolution")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Phase 1: Accumulate per-cell statistics and transitions
    # ============================================================
    print("\n=== Phase 1: Accumulating statistics ===")

    # Accumulators: {cell_id: [sog_sum, sog_sq_sum, temp_sum, wind_sum, prmsl_sum, vis_sum, count]}
    cell_accum = defaultdict(lambda: np.zeros(7, dtype=np.float64))
    trans_counter = defaultdict(int)
    last_vid, last_cell = None, None
    total_rows = 0
    weather_cols = ['temp', 'wind_speed', 'prmsl', 'visibility']

    for i, chunk in enumerate(pd.read_csv(args.data_path, chunksize=args.chunk_size)):
        total_rows += len(chunk)

        # Fill NaN weather
        for col in weather_cols:
            if col in chunk.columns:
                chunk[col] = chunk[col].fillna(0)
        chunk = chunk.dropna(subset=['lat', 'lon', 'sog'])

        # Map to cells
        cells = latlon_to_raw_cell(chunk['lat'].values, chunk['lon'].values,
                                   cell_size, num_lat, num_lon)
        chunk = chunk.copy()
        chunk['cell'] = cells
        chunk['sog_sq'] = chunk['sog'].values ** 2

        # Per-cell statistics via groupby
        grouped = chunk.groupby('cell').agg(
            sog_sum=('sog', 'sum'),
            sog_sq_sum=('sog_sq', 'sum'),
            temp_sum=('temp', 'sum'),
            wind_sum=('wind_speed', 'sum'),
            prmsl_sum=('prmsl', 'sum'),
            vis_sum=('visibility', 'sum'),
            count=('sog', 'count')
        )
        for cell_id, row in grouped.iterrows():
            cell_accum[int(cell_id)] += row.values

        # Transitions within same voyage
        vid = chunk['voyage_id'].values

        # Handle cross-chunk boundary
        if last_vid is not None and len(vid) > 0:
            if vid[0] == last_vid and cells[0] != last_cell:
                trans_counter[(int(last_cell), int(cells[0]))] += 1

        # Intra-chunk transitions (vectorized)
        same_voyage = vid[1:] == vid[:-1]
        diff_cell = cells[1:] != cells[:-1]
        mask = same_voyage & diff_cell
        if mask.any():
            src = cells[:-1][mask]
            dst = cells[1:][mask]
            pairs = np.stack([src, dst], axis=1)
            unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
            for (s, d), c in zip(unique_pairs, counts):
                trans_counter[(int(s), int(d))] += int(c)

        if len(vid) > 0:
            last_vid = vid[-1]
            last_cell = cells[-1]

        if (i + 1) % 5 == 0:
            print(f"  Chunk {i+1}: {total_rows:,} rows processed")

    print(f"Total: {total_rows:,} rows")
    print(f"Unique cells seen: {len(cell_accum)}")
    print(f"Unique transitions: {len(trans_counter)}")

    # ============================================================
    # Phase 2: Build compact cell mapping
    # ============================================================
    print("\n=== Phase 2: Filtering active cells ===")

    active_raw_ids = sorted([c for c, v in cell_accum.items() if v[6] >= MIN_CELL_COUNT])
    cell_to_compact = {c: i for i, c in enumerate(active_raw_ids)}
    UNKNOWN_NODE = len(active_raw_ids)
    N = len(active_raw_ids) + 1  # +1 for unknown

    print(f"Active cells (count≥{MIN_CELL_COUNT}): {len(active_raw_ids)}")
    print(f"Total graph nodes: {N}")

    # ============================================================
    # Phase 3: Node features
    # ============================================================
    print("\n=== Phase 3: Building node features ===")

    # Features: mean_sog, std_sog, mean_temp, mean_wind, mean_prmsl, mean_vis,
    #           log_count, center_lat_norm, center_lon_norm
    node_features = np.zeros((N, 9), dtype=np.float32)

    for raw_id, compact_id in cell_to_compact.items():
        data = cell_accum[raw_id]
        count = data[6]
        mean_sog = data[0] / count
        var_sog = max(0, data[1] / count - mean_sog ** 2)

        node_features[compact_id, 0] = mean_sog
        node_features[compact_id, 1] = np.sqrt(var_sog)
        node_features[compact_id, 2] = data[2] / count  # mean_temp
        node_features[compact_id, 3] = data[3] / count  # mean_wind
        node_features[compact_id, 4] = data[4] / count  # mean_prmsl
        node_features[compact_id, 5] = data[5] / count  # mean_vis
        node_features[compact_id, 6] = np.log1p(count)  # log_count

        # Cell center (normalized to [0, 1])
        lat_idx = raw_id // num_lon
        lon_idx = raw_id % num_lon
        node_features[compact_id, 7] = (lat_idx + 0.5) / num_lat
        node_features[compact_id, 8] = (lon_idx + 0.5) / num_lon

    # Unknown node: average of all active nodes
    if UNKNOWN_NODE > 0 and UNKNOWN_NODE < N:
        node_features[UNKNOWN_NODE] = node_features[:UNKNOWN_NODE].mean(axis=0)

    # Z-score normalize
    feat_mean = node_features.mean(axis=0)
    feat_std = node_features.std(axis=0) + 1e-8
    node_features_norm = ((node_features - feat_mean) / feat_std).astype(np.float32)

    print(f"Node features shape: {node_features_norm.shape}")

    # ============================================================
    # Phase 4: Adjacency matrix
    # ============================================================
    print("\n=== Phase 4: Building adjacency matrix ===")

    adj = np.zeros((N, N), dtype=np.float32)

    # Transition-based edges
    trans_edges = 0
    for (s, d), cnt in trans_counter.items():
        if s in cell_to_compact and d in cell_to_compact:
            si, di = cell_to_compact[s], cell_to_compact[d]
            adj[si, di] += cnt
            adj[di, si] += cnt  # undirected
            trans_edges += 1

    # Geographic neighbor edges (8-connected) with weight 1
    geo_edges = 0
    for raw_id in active_raw_ids:
        lat_idx = raw_id // num_lon
        lon_idx = raw_id % num_lon
        for dlat in [-1, 0, 1]:
            for dlon in [-1, 0, 1]:
                if dlat == 0 and dlon == 0:
                    continue
                nlat = lat_idx + dlat
                nlon = (lon_idx + dlon) % num_lon  # wrap longitude
                if 0 <= nlat < num_lat:
                    neighbor_raw = nlat * num_lon + nlon
                    if neighbor_raw in cell_to_compact:
                        si = cell_to_compact[raw_id]
                        di = cell_to_compact[neighbor_raw]
                        if adj[si, di] == 0:
                            adj[si, di] = 1
                            adj[di, si] = 1
                            geo_edges += 1

    print(f"Transition edges: {trans_edges}, Geographic edges: {geo_edges}")
    print(f"Non-zero adjacency entries: {(adj > 0).sum()}")

    # Self-loops + symmetric normalization: D^(-1/2)(A+I)D^(-1/2)
    adj += np.eye(N, dtype=np.float32)
    D = adj.sum(axis=1)
    D_inv_sqrt = np.zeros_like(D)
    D_inv_sqrt[D > 0] = 1.0 / np.sqrt(D[D > 0])
    adj_norm = (D_inv_sqrt[:, None] * adj * D_inv_sqrt[None, :]).astype(np.float32)

    # ============================================================
    # Phase 5: Save graph
    # ============================================================
    print("\n=== Phase 5: Saving graph ===")

    np.save(output_dir / 'adj_normalized.npy', adj_norm)
    np.save(output_dir / 'node_features.npy', node_features_norm)
    np.save(output_dir / 'node_feat_mean.npy', feat_mean)
    np.save(output_dir / 'node_feat_std.npy', feat_std)

    with open(output_dir / 'cell_to_compact.json', 'w') as f:
        json.dump({str(k): v for k, v in cell_to_compact.items()}, f)

    meta = {
        'cell_size': cell_size,
        'lat_min': LAT_MIN, 'lat_max': LAT_MAX,
        'lon_min': LON_MIN, 'lon_max': LON_MAX,
        'num_lat_bins': num_lat, 'num_lon_bins': num_lon,
        'num_active_cells': len(active_raw_ids),
        'num_nodes': N,
        'unknown_node': UNKNOWN_NODE,
        'node_feature_dim': 9,
        'min_cell_count': MIN_CELL_COUNT,
    }
    with open(output_dir / 'graph_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"Graph saved to {output_dir}")

    # ============================================================
    # Phase 6: Compute cell_ids for cached sequence data
    # ============================================================
    print("\n=== Phase 6: Computing cell_ids for cached data ===")

    cache_dir = Path(args.cache_dir)
    cell_ids_out_dir = Path(args.cell_ids_dir) if args.cell_ids_dir else cache_dir
    cell_ids_out_dir.mkdir(parents=True, exist_ok=True)

    norm = np.load(args.norm_path, allow_pickle=True)
    fmin = norm['feature_min']
    fmax = norm['feature_max']
    counts = np.load(cache_dir / 'actual_counts.npy', allow_pickle=True).item()

    # Build lookup table: raw_cell_id -> compact_cell_id
    lookup = np.full(max_raw_id, UNKNOWN_NODE, dtype=np.int32)
    for raw_id, compact_id in cell_to_compact.items():
        if 0 <= raw_id < max_raw_id:
            lookup[raw_id] = compact_id

    for split in ['train', 'val', 'test']:
        n = counts[split]
        X = np.load(cache_dir / f'X_{split}.npy', mmap_mode='r')
        cell_ids = np.zeros((n, 48), dtype=np.int32)

        CHUNK = 500_000
        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)
            chunk_data = X[start:end]
            # Denormalize lat (feature 0) and lon (feature 1)
            lat = chunk_data[:, :, 0].astype(np.float64) * (fmax[0] - fmin[0]) + fmin[0]
            lon = chunk_data[:, :, 1].astype(np.float64) * (fmax[1] - fmin[1]) + fmin[1]
            raw = latlon_to_raw_cell(lat, lon, cell_size, num_lat, num_lon)
            raw = np.clip(raw, 0, max_raw_id - 1)
            cell_ids[start:end] = lookup[raw]

        save_path = cell_ids_out_dir / f'cell_ids_{split}.npy'
        np.save(save_path, cell_ids)
        print(f"  {split}: {n:,} samples → {save_path}")

    # Summary
    print("\n=== Summary ===")
    print(f"Graph: {N} nodes, {(adj_norm > 0).sum()} non-zero adjacency entries")
    print(f"Node features: {node_features_norm.shape}")

    for split in ['train', 'val', 'test']:
        n = counts[split]
        cids = np.load(cell_ids_out_dir / f'cell_ids_{split}.npy')[:n]
        unknown_pct = (cids == UNKNOWN_NODE).mean() * 100
        unique_cells = len(np.unique(cids))
        print(f"  {split}: {unknown_pct:.2f}% unknown cells, {unique_cells} unique cells used")

    print("\nDone!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build maritime route graph')
    parser.add_argument('--data_path', default='output/processed/processed_voyages.csv')
    parser.add_argument('--cache_dir',
                        default='output/cache_sequences/seq48_label24_pred1_mv150000_ms50000000')
    parser.add_argument('--norm_path', default='output/norm_params.npz')
    parser.add_argument('--output_dir', default='output/graph')
    parser.add_argument('--cell_ids_dir', default=None,
                        help='Directory to save cell_ids_*.npy. Defaults to cache_dir.')
    parser.add_argument('--chunk_size', type=int, default=2_000_000)
    parser.add_argument('--cell_size', type=float, default=2.0)
    args = parser.parse_args()
    build_graph(args)
