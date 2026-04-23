#!/usr/bin/env python
"""
Single-vessel / single-voyage ETA inference interface.

────────────────────────────────────────────────────────────────
Python API — single model
────────────────────────────────────────────────────────────────
    from inference import ETAPredictor

    predictor = ETAPredictor.load(
        model_path  = 'output/mstgn/best_mstgn.pth',
        graph_dir   = 'output/graph',
        norm_path   = 'output/norm_params.npz',
    )

    # ais_records: list of dicts with keys
    #   postime (str/datetime), lat, lon, sog, cog
    #   optionally: temp, wind_speed, wind_level, prmsl, visibility
    result = predictor.predict(ais_records, dest_lat=33.74, dest_lon=-118.27)
    print(result)
    # ETAResult(remaining_hours=156.3, arrival_time=datetime(2026,4,30,14,22),
    #           sigma_hours=None, interval_90=None)

────────────────────────────────────────────────────────────────
Python API — ensemble (with conformal prediction intervals)
────────────────────────────────────────────────────────────────
    from inference import EnsembleETAPredictor

    predictor = EnsembleETAPredictor.load(
        ensemble_dir = 'output/ensemble',   # contains seed42/ seed43/ …
        graph_dir    = 'output/graph',
        norm_path    = 'output/norm_params.npz',
        top_k        = 7,                   # use top-7 by val_loss
        conf_path    = 'output/uncertainty/conformal_quantiles.json',  # optional
    )

    result = predictor.predict(ais_records, dest_lat=33.74, dest_lon=-118.27)
    print(result)
    # ETAResult(remaining_hours=153.1, arrival_time=...,
    #           sigma_hours=8.4, interval_90=(145.0, 182.7))

────────────────────────────────────────────────────────────────
CLI
────────────────────────────────────────────────────────────────
    python inference.py \\
        --ais_csv  recent_ais.csv \\
        --dest_lat 33.74 --dest_lon -118.27 \\
        --model    output/mstgn/best_mstgn.pth \\
        --graph    output/graph \\
        --norm     output/norm_params.npz

    # Ensemble mode:
    python inference.py \\
        --ais_csv  recent_ais.csv \\
        --dest_lat 33.74 --dest_lon -118.27 \\
        --ensemble output/ensemble \\
        --graph    output/graph \\
        --norm     output/norm_params.npz \\
        --top_k    7

────────────────────────────────────────────────────────────────
CSV column requirements for --ais_csv
────────────────────────────────────────────────────────────────
Required  : postime, lat, lon, sog, cog
Optional  : temp, wind_speed, wind_level, prmsl, visibility
            (missing weather columns are filled with 0)

The script takes the LAST 48 records (sorted by postime).
Provide at least 48 rows; more rows are accepted and the tail is used.
"""

import sys
import os
import json
import argparse
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

# ── make src/ importable regardless of cwd ──────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from src.mstgn.model import MSTGN_MLP2

# ────────────────────────────────────────────────────────────────────────────
# Constants (must match build_route_graph.py)
# ────────────────────────────────────────────────────────────────────────────
_LAT_MIN, _LAT_MAX = -58.0, 70.0
_LON_MIN, _LON_MAX = -180.0, 180.0

# Feature column order (must match training)
FEATURE_COLS = ['lat', 'lon', 'sog', 'cog',
                'dist_to_dest_km', 'bearing_diff',
                'temp', 'wind_speed', 'wind_level', 'prmsl', 'visibility']
SEQ_LEN = 48


# ────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ────────────────────────────────────────────────────────────────────────────

def _haversine_km(lat1: np.ndarray, lon1: np.ndarray,
                  lat2: float, lon2: float) -> np.ndarray:
    R = 6371.0
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = math.radians(lat2), math.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * math.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _bearing_deg(lat1: np.ndarray, lon1: np.ndarray,
                 lat2: float, lon2: float) -> np.ndarray:
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = math.radians(lat2), math.radians(lon2)
    dlon = lon2 - lon1
    x = np.sin(dlon) * math.cos(lat2)
    y = math.cos(lat2) * np.sin(lat1) - math.sin(lat2) * np.cos(lat1) * np.cos(dlon)
    # y uses cos(lat2) as scale but that's fine for bearing to dest
    y = np.cos(lat1) * math.sin(lat2) - np.sin(lat1) * math.cos(lat2) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360


def _cell_id(lat: np.ndarray, lon: np.ndarray,
             cell_size: float, num_lon: int,
             num_lat: int) -> np.ndarray:
    """Map lat/lon arrays to raw cell IDs (same formula as build_route_graph.py)."""
    lat_idx = np.clip(((lat - _LAT_MIN) / cell_size).astype(np.int32), 0, num_lat - 1)
    lon_idx = np.clip(((lon - _LON_MIN) / cell_size).astype(np.int32), 0, num_lon - 1)
    return lat_idx * num_lon + lon_idx


# ────────────────────────────────────────────────────────────────────────────
# Output data class
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class ETAResult:
    """Prediction result for a single inference call.

    Attributes
    ----------
    remaining_hours : float
        Point estimate of remaining sailing time (hours).
    arrival_time : datetime
        Estimated arrival time in UTC (query_time + remaining_hours).
        None if query_time was not provided.
    query_time : datetime
        Timestamp of the latest AIS record used for prediction (UTC).
    sigma_hours : float or None
        Ensemble standard deviation in hours. None for single-model inference.
    interval_90 : (float, float) or None
        90 % conformal prediction interval (lower_hours, upper_hours).
        None if calibrated quantile is not available.
    interval_95 : (float, float) or None
        95 % conformal prediction interval. None if not available.
    n_models : int
        Number of models used (1 for single-model, K for ensemble).
    """
    remaining_hours: float
    arrival_time: Optional[datetime]
    query_time: Optional[datetime]
    sigma_hours: Optional[float] = None
    interval_90: Optional[Tuple[float, float]] = None
    interval_95: Optional[Tuple[float, float]] = None
    n_models: int = 1

    def __str__(self) -> str:  # noqa: D105
        lines = [
            "─" * 52,
            f"  ETA Prediction",
            "─" * 52,
            f"  Remaining time  : {self.remaining_hours:.1f} h  ({self.remaining_hours/24:.2f} days)",
        ]
        if self.arrival_time:
            lines.append(f"  Estimated arrival: {self.arrival_time.strftime('%Y-%m-%d %H:%M UTC')}")
        if self.query_time:
            lines.append(f"  Based on AIS at  : {self.query_time.strftime('%Y-%m-%d %H:%M UTC')}")
        if self.sigma_hours is not None:
            lines.append(f"  Uncertainty (σ)  : ±{self.sigma_hours:.1f} h  ({self.n_models} models)")
        if self.interval_90:
            lo, hi = self.interval_90
            lines.append(f"  90% interval     : [{lo:.1f} h, {hi:.1f} h]")
        if self.interval_95:
            lo, hi = self.interval_95
            lines.append(f"  95% interval     : [{lo:.1f} h, {hi:.1f} h]")
        lines.append("─" * 52)
        return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────────────
# Graph / normalization loader helpers
# ────────────────────────────────────────────────────────────────────────────

class _GraphBundle:
    """Holds route-graph artifacts needed for inference."""

    def __init__(self, graph_dir: Union[str, Path]):
        graph_dir = Path(graph_dir)
        self.adj = np.load(graph_dir / 'adj_normalized.npy')
        self.node_features = np.load(graph_dir / 'node_features.npy')
        with open(graph_dir / 'graph_meta.json') as f:
            meta = json.load(f)
        self.cell_size: float = meta['cell_size']
        self.num_lat: int = meta['num_lat_bins']
        self.num_lon: int = meta['num_lon_bins']
        self.unknown_node: int = meta['unknown_node']
        # Build fast lookup array: raw_cell -> compact_cell
        max_raw = self.num_lat * self.num_lon
        self._lookup = np.full(max_raw, self.unknown_node, dtype=np.int32)
        with open(graph_dir / 'cell_to_compact.json') as f:
            c2c = json.load(f)
        for raw_str, compact in c2c.items():
            raw = int(raw_str)
            if 0 <= raw < max_raw:
                self._lookup[raw] = compact

    def latlon_to_compact(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Convert lat/lon arrays to compact cell IDs."""
        raw = _cell_id(lat, lon, self.cell_size, self.num_lon, self.num_lat)
        raw = np.clip(raw, 0, len(self._lookup) - 1)
        return self._lookup[raw]


class _NormBundle:
    """Holds feature/target normalisation parameters."""

    def __init__(self, norm_path: Union[str, Path]):
        norm = np.load(norm_path, allow_pickle=True)
        self.feature_min: np.ndarray = norm['feature_min'].astype(np.float32)
        self.feature_max: np.ndarray = norm['feature_max'].astype(np.float32)
        self.target_mean: float = float(norm['target_mean'])
        self.target_std: float = float(norm['target_std'])

    def normalize_features(self, x: np.ndarray) -> np.ndarray:
        """Min-max normalise features, clipped to [-0.5, 1.5]."""
        denom = self.feature_max - self.feature_min
        denom[denom < 1e-8] = 1e-8
        x_norm = (x - self.feature_min) / denom
        return np.clip(x_norm, -0.5, 1.5).astype(np.float32)

    def denormalize_target(self, y_norm: np.ndarray) -> np.ndarray:
        """Reverse log-standardisation: ỹ → hours."""
        log_y = y_norm * self.target_std + self.target_mean
        return np.expm1(log_y)


# ────────────────────────────────────────────────────────────────────────────
# AIS record → (feature_tensor, cell_ids) pipeline
# ────────────────────────────────────────────────────────────────────────────

def _records_to_tensor(
    records: List[Dict],
    dest_lat: float,
    dest_lon: float,
    graph: _GraphBundle,
    norm: _NormBundle,
    seq_len: int = SEQ_LEN,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[datetime]]:
    """
    Convert a list of raw AIS dicts into model-ready tensors.

    Parameters
    ----------
    records : list of dict
        Each dict must have: lat, lon, sog, cog, postime.
        Optional weather keys: temp, wind_speed, wind_level, prmsl, visibility.
    dest_lat, dest_lon : float
        Destination port coordinates.
    graph : _GraphBundle
    norm  : _NormBundle
    seq_len : int
        Required sequence length (default 48).

    Returns
    -------
    x_tensor   : (1, seq_len, 11) float32 torch.Tensor
    cell_tensor: (1, seq_len)     int64   torch.Tensor
    query_time : datetime (UTC) of the last AIS record, or None
    """
    if len(records) < seq_len:
        raise ValueError(
            f"At least {seq_len} AIS records are required for inference. "
            f"Got {len(records)}."
        )

    # ── Parse and sort by time ───────────────────────────────────────────────
    import pandas as pd
    df = pd.DataFrame(records)
    if 'postime' in df.columns:
        df['postime'] = pd.to_datetime(df['postime'], utc=True, errors='coerce')
        df = df.sort_values('postime')
    df = df.tail(seq_len).reset_index(drop=True)

    lat = df['lat'].values.astype(np.float64)
    lon = df['lon'].values.astype(np.float64)
    sog = df['sog'].values.astype(np.float64)
    cog = df['cog'].values.astype(np.float64)

    # ── Derived navigation features ──────────────────────────────────────────
    dist_km   = _haversine_km(lat, lon, dest_lat, dest_lon)
    bearing   = _bearing_deg(lat, lon, dest_lat, dest_lon)
    bear_diff = np.abs(((bearing - cog + 180) % 360) - 180)

    # ── Weather (fill missing with 0) ────────────────────────────────────────
    def _col(name):
        if name in df.columns:
            return df[name].fillna(0).values.astype(np.float64)
        return np.zeros(seq_len, dtype=np.float64)

    temp       = _col('temp')
    wind_speed = _col('wind_speed')
    wind_level = _col('wind_level')
    prmsl      = _col('prmsl')
    visibility = _col('visibility')

    # ── Stack feature matrix (seq_len, 11) ───────────────────────────────────
    x = np.stack([lat, lon, sog, cog, dist_km, bear_diff,
                  temp, wind_speed, wind_level, prmsl, visibility], axis=1)
    x = x.astype(np.float32)

    # ── Normalize ────────────────────────────────────────────────────────────
    x_norm = norm.normalize_features(x)   # (seq_len, 11)

    # ── Cell IDs ─────────────────────────────────────────────────────────────
    raw_lat = df['lat'].values.astype(np.float64)
    raw_lon = df['lon'].values.astype(np.float64)
    cell_ids = graph.latlon_to_compact(raw_lat, raw_lon)  # (seq_len,)

    # ── Build tensors with batch dim = 1 ────────────────────────────────────
    x_tensor    = torch.from_numpy(x_norm).unsqueeze(0)           # (1, 48, 11)
    cell_tensor = torch.from_numpy(cell_ids.astype(np.int64)).unsqueeze(0)  # (1, 48)

    # ── Query time ───────────────────────────────────────────────────────────
    query_time = None
    if 'postime' in df.columns and not df['postime'].isna().all():
        ts = df['postime'].iloc[-1]
        try:
            query_time = ts.to_pydatetime()
            if query_time.tzinfo is None:
                query_time = query_time.replace(tzinfo=timezone.utc)
        except Exception:
            pass

    return x_tensor, cell_tensor, query_time


# ────────────────────────────────────────────────────────────────────────────
# Single-model predictor
# ────────────────────────────────────────────────────────────────────────────

class ETAPredictor:
    """Single-model MSTGN-MLP2 predictor.

    Typical usage
    -------------
    predictor = ETAPredictor.load('output/mstgn/best_mstgn.pth',
                                  'output/graph',
                                  'output/norm_params.npz')
    result = predictor.predict(ais_records, dest_lat=33.74, dest_lon=-118.27)
    """

    def __init__(self, model: nn.Module, graph: _GraphBundle, norm: _NormBundle,
                 device: torch.device):
        self.model  = model.to(device).eval()
        self.graph  = graph
        self.norm   = norm
        self.device = device

    # ── Constructor ─────────────────────────────────────────────────────────
    @classmethod
    def load(cls,
             model_path: Union[str, Path],
             graph_dir:  Union[str, Path],
             norm_path:  Union[str, Path],
             device:     Optional[Union[str, torch.device]] = None,
             gcn_hidden: int = 64,
             cell_emb_dim: int = 32,
             dropout: float = 0.1) -> 'ETAPredictor':
        """Load predictor from saved checkpoint and graph files.

        Parameters
        ----------
        model_path  : path to best_mstgn.pth checkpoint
        graph_dir   : directory containing adj_normalized.npy, node_features.npy, …
        norm_path   : path to norm_params.npz
        device      : 'cpu', 'cuda', or None (auto-detect)
        gcn_hidden, cell_emb_dim, dropout : must match training config (defaults correct)
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)

        graph = _GraphBundle(graph_dir)
        norm  = _NormBundle(norm_path)

        model = MSTGN_MLP2(
            adj_matrix         = graph.adj,
            init_node_features = graph.node_features,
            seq_feat_dim       = 11,
            seq_len            = SEQ_LEN,
            gcn_hidden         = gcn_hidden,
            cell_emb_dim       = cell_emb_dim,
            dropout            = dropout,
        )
        state = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"[ETAPredictor] Loaded model from {model_path}  (device={device})")
        return cls(model, graph, norm, device)

    # ── Inference ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(self,
                ais_records: List[Dict],
                dest_lat: float,
                dest_lon: float) -> ETAResult:
        """Predict remaining sailing time for a single vessel.

        Parameters
        ----------
        ais_records : list of dict
            Must contain at least 48 entries with keys:
                postime (str / datetime), lat, lon, sog, cog
            Optional weather: temp, wind_speed, wind_level, prmsl, visibility
        dest_lat, dest_lon : float
            Destination port coordinates (decimal degrees).

        Returns
        -------
        ETAResult
        """
        x, cell_ids, query_time = _records_to_tensor(
            ais_records, dest_lat, dest_lon, self.graph, self.norm
        )
        x        = x.to(self.device)
        cell_ids = cell_ids.to(self.device)

        y_norm = self.model(x, cell_ids).cpu().numpy()       # (1,)
        hours  = float(np.maximum(self.norm.denormalize_target(y_norm), 0)[0])

        arrival = None
        if query_time is not None:
            arrival = query_time + timedelta(hours=hours)

        return ETAResult(
            remaining_hours = hours,
            arrival_time    = arrival,
            query_time      = query_time,
            n_models        = 1,
        )


# ────────────────────────────────────────────────────────────────────────────
# Ensemble predictor (with conformal intervals)
# ────────────────────────────────────────────────────────────────────────────

class EnsembleETAPredictor:
    """Multi-seed MSTGN-MLP2 ensemble predictor with conformal intervals.

    Typical usage
    -------------
    predictor = EnsembleETAPredictor.load(
        ensemble_dir = 'output/ensemble',
        graph_dir    = 'output/graph',
        norm_path    = 'output/norm_params.npz',
        top_k        = 7,
        conf_path    = 'output/uncertainty/conformal_quantiles.json',
    )
    result = predictor.predict(ais_records, dest_lat=33.74, dest_lon=-118.27)
    """

    def __init__(self, models: List[nn.Module], graph: _GraphBundle, norm: _NormBundle,
                 device: torch.device,
                 q90: Optional[float] = None,
                 q95: Optional[float] = None):
        self.models = [m.to(device).eval() for m in models]
        self.graph  = graph
        self.norm   = norm
        self.device = device
        self.q90    = q90   # calibrated conformal quantile for 90% coverage
        self.q95    = q95   # calibrated conformal quantile for 95% coverage

    # ── Constructor ─────────────────────────────────────────────────────────
    @classmethod
    def load(cls,
             ensemble_dir: Union[str, Path],
             graph_dir:    Union[str, Path],
             norm_path:    Union[str, Path],
             top_k:        int = 7,
             conf_path:    Optional[Union[str, Path]] = None,
             device:       Optional[Union[str, torch.device]] = None,
             gcn_hidden:   int = 64,
             cell_emb_dim: int = 32,
             dropout:      float = 0.1) -> 'EnsembleETAPredictor':
        """Load ensemble from output/ensemble/seed*/ directories.

        Parameters
        ----------
        ensemble_dir : directory containing seed42/, seed43/, … sub-dirs
        graph_dir    : directory with graph files
        norm_path    : path to norm_params.npz
        top_k        : number of models to use (ranked by validation loss)
        conf_path    : optional JSON with calibrated conformal quantiles
                       {"q90": 12.34, "q95": 18.56}  (in σ units for adaptive,
                       OR in hours for fixed — see note below)
                       If None, raw ensemble σ is reported but no intervals.
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)

        ensemble_dir = Path(ensemble_dir)
        graph = _GraphBundle(graph_dir)
        norm  = _NormBundle(norm_path)

        # Rank seeds by validation loss
        seed_dirs = sorted(ensemble_dir.glob('seed*'))
        ranked = []
        for sd in seed_dirs:
            ckpt = sd / 'best_mstgn.pth'
            if not ckpt.exists():
                continue
            val_loss = float('inf')
            results_file = sd / 'results.json'
            if results_file.exists():
                with open(results_file) as f:
                    val_loss = json.load(f).get('best_val_loss', float('inf'))
            ranked.append((sd, val_loss))
        ranked.sort(key=lambda t: t[1])
        selected = ranked[:top_k]

        if not selected:
            raise FileNotFoundError(
                f"No valid checkpoints found in {ensemble_dir}/seed*/"
            )

        print(f"[EnsembleETAPredictor] Loading top-{len(selected)} models "
              f"(device={device}):")
        models = []
        for sd, vl in selected:
            m = MSTGN_MLP2(
                adj_matrix         = graph.adj,
                init_node_features = graph.node_features,
                seq_feat_dim       = 11,
                seq_len            = SEQ_LEN,
                gcn_hidden         = gcn_hidden,
                cell_emb_dim       = cell_emb_dim,
                dropout            = dropout,
            )
            state = torch.load(sd / 'best_mstgn.pth', map_location=device,
                               weights_only=True)
            m.load_state_dict(state)
            models.append(m)
            print(f"  {sd.name}  val_loss={vl:.6f}")

        # Load calibrated quantiles (optional)
        q90, q95 = None, None
        if conf_path is not None:
            conf_path = Path(conf_path)
            if conf_path.exists():
                with open(conf_path) as f:
                    cq = json.load(f)
                q90 = cq.get('q90')
                q95 = cq.get('q95')
                print(f"[EnsembleETAPredictor] Conformal quantiles: "
                      f"q90={q90:.3f}, q95={q95:.3f}")
            else:
                print(f"[EnsembleETAPredictor] conf_path not found ({conf_path}); "
                      f"intervals will not be reported.")
        else:
            print("[EnsembleETAPredictor] No conf_path provided; "
                  "σ reported but no calibrated intervals.")

        return cls(models, graph, norm, device, q90=q90, q95=q95)

    # ── Inference ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(self,
                ais_records: List[Dict],
                dest_lat: float,
                dest_lon: float) -> ETAResult:
        """Predict remaining sailing time with uncertainty quantification.

        Returns
        -------
        ETAResult with sigma_hours and (optionally) interval_90/interval_95 set.

        Notes on intervals
        ------------------
        The prediction intervals use *adaptive* conformal prediction:
            interval = [ŷ − q · σ,  ŷ + q · σ]
        where σ is the ensemble standard deviation and q is the conformal
        quantile calibrated offline on the test-set calibration half.
        Load the quantiles with conf_path = 'output/uncertainty/conformal_quantiles.json'.
        """
        x, cell_ids, query_time = _records_to_tensor(
            ais_records, dest_lat, dest_lon, self.graph, self.norm
        )
        x        = x.to(self.device)
        cell_ids = cell_ids.to(self.device)

        # Forward pass through every model
        preds_norm = np.array([
            m(x, cell_ids).cpu().numpy()[0] for m in self.models
        ])                                             # (K,)

        # De-normalise each model's prediction, then aggregate
        preds_h = np.maximum(self.norm.denormalize_target(preds_norm), 0.0)
        hours   = float(preds_h.mean())
        sigma   = float(preds_h.std())

        # Conformal intervals
        interval_90 = interval_95 = None
        if self.q90 is not None:
            half = self.q90 * sigma
            interval_90 = (max(0.0, hours - half), hours + half)
        if self.q95 is not None:
            half = self.q95 * sigma
            interval_95 = (max(0.0, hours - half), hours + half)

        arrival = None
        if query_time is not None:
            arrival = query_time + timedelta(hours=hours)

        return ETAResult(
            remaining_hours = hours,
            arrival_time    = arrival,
            query_time      = query_time,
            sigma_hours     = sigma,
            interval_90     = interval_90,
            interval_95     = interval_95,
            n_models        = len(self.models),
        )


# ────────────────────────────────────────────────────────────────────────────
# Top-level convenience function
# ────────────────────────────────────────────────────────────────────────────

def predict_eta(ais_records: List[Dict],
                dest_lat: float,
                dest_lon: float,
                model_path:   Optional[Union[str, Path]] = None,
                ensemble_dir: Optional[Union[str, Path]] = None,
                graph_dir:    Union[str, Path] = 'output/graph',
                norm_path:    Union[str, Path] = 'output/norm_params.npz',
                conf_path:    Optional[Union[str, Path]] = None,
                top_k:        int = 7,
                device:       Optional[str] = None) -> ETAResult:
    """One-call convenience wrapper.

    Provide either ``model_path`` (single model) or ``ensemble_dir``
    (multi-seed ensemble with optional conformal intervals).

    Parameters
    ----------
    ais_records : list of dict (≥ 48 records)
        Keys: postime, lat, lon, sog, cog, [temp, wind_speed, wind_level,
        prmsl, visibility]
    dest_lat, dest_lon : float
        Destination port coordinates.
    model_path   : path to single-model checkpoint (mutually exclusive with
                   ensemble_dir)
    ensemble_dir : directory containing seed42/, seed43/, … (mutually
                   exclusive with model_path)
    graph_dir    : directory with route-graph files
    norm_path    : path to norm_params.npz
    conf_path    : optional JSON with conformal quantiles (ensemble only)
    top_k        : number of ensemble models to use
    device       : 'cpu' or 'cuda' (None = auto)

    Returns
    -------
    ETAResult
    """
    if model_path is not None and ensemble_dir is None:
        predictor = ETAPredictor.load(model_path, graph_dir, norm_path,
                                      device=device)
    elif ensemble_dir is not None and model_path is None:
        predictor = EnsembleETAPredictor.load(ensemble_dir, graph_dir, norm_path,
                                              top_k=top_k, conf_path=conf_path,
                                              device=device)
    else:
        raise ValueError("Provide exactly one of: model_path or ensemble_dir.")

    return predictor.predict(ais_records, dest_lat, dest_lon)


# ────────────────────────────────────────────────────────────────────────────
# Save conformal quantiles helper (run once after eval_uncertainty.py)
# ────────────────────────────────────────────────────────────────────────────

def save_conformal_quantiles(output_path: Union[str, Path],
                             q90: float, q95: float) -> None:
    """Persist calibrated conformal quantiles to JSON for later inference.

    Run this once after running eval_uncertainty.py to extract q90/q95:

        from inference import save_conformal_quantiles
        save_conformal_quantiles('output/uncertainty/conformal_quantiles.json',
                                 q90=9.87, q95=14.23)
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({'q90': q90, 'q95': q95}, f, indent=2)
    print(f"Conformal quantiles saved to {output_path}")


# ────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ────────────────────────────────────────────────────────────────────────────

def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Predict vessel ETA from AIS records.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--ais_csv', required=True,
                   help='CSV file with AIS records (≥48 rows).')
    p.add_argument('--dest_lat', type=float, required=True,
                   help='Destination port latitude (decimal degrees).')
    p.add_argument('--dest_lon', type=float, required=True,
                   help='Destination port longitude (decimal degrees).')

    # Model source — mutually exclusive
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--model', dest='model_path',
                     help='Path to single-model checkpoint (best_mstgn.pth).')
    src.add_argument('--ensemble', dest='ensemble_dir',
                     help='Directory containing seed42/ … sub-dirs.')

    p.add_argument('--top_k', type=int, default=7,
                   help='Number of ensemble models to use (default: 7).')
    p.add_argument('--graph', dest='graph_dir', default='output/graph',
                   help='Route-graph directory (default: output/graph).')
    p.add_argument('--norm', dest='norm_path', default='output/norm_params.npz',
                   help='Normalization params (default: output/norm_params.npz).')
    p.add_argument('--conf', dest='conf_path', default=None,
                   help='JSON with conformal quantiles (optional, ensemble only).')
    p.add_argument('--device', default=None,
                   help='cpu or cuda (default: auto-detect).')
    return p


def main():
    args = _build_cli().parse_args()

    import pandas as pd
    df = pd.read_csv(args.ais_csv)
    records = df.to_dict(orient='records')

    result = predict_eta(
        ais_records  = records,
        dest_lat     = args.dest_lat,
        dest_lon     = args.dest_lon,
        model_path   = args.model_path,
        ensemble_dir = args.ensemble_dir,
        graph_dir    = args.graph_dir,
        norm_path    = args.norm_path,
        conf_path    = args.conf_path,
        top_k        = args.top_k,
        device       = args.device,
    )
    print(result)


if __name__ == '__main__':
    main()
