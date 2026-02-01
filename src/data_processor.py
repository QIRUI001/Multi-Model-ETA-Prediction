"""
Data preprocessing and dataset for ETA prediction using Informer-TP.

Processes AIS data containing ship trajectory information to predict ETA.
Supports batch processing to avoid OOM.
"""

import os
import glob
import gc
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, List, Optional, Generator, Iterator
from sklearn.preprocessing import StandardScaler
import pickle

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset


class ETADataProcessor:
    """
    Process AIS data for ETA prediction.
    
    Input data format (from CSV):
    - postime: Position timestamp
    - eta: Estimated time of arrival (ship-reported)
    - lon, lat: Longitude and Latitude
    - sog: Speed over ground (knots)
    - cog: Course over ground (degrees)
    - Additional weather and navigation features
    
    Output:
    - Features: [lat, lon, sog, cog] normalized
    - Target: Remaining time to arrival (hours)
    - Time features: [minute, hour, day, weekday, month] normalized to [-0.5, 0.5]
    """
    
    def __init__(self, data_dir: str, seq_len: int = 48, label_len: int = 24, pred_len: int = 1,
                 norm_type: str = 'minmax'):
        """
        Args:
            data_dir: Directory containing AIS CSV files
            seq_len: Input sequence length
            label_len: Start token length (overlap)
            pred_len: Prediction length
            norm_type: Normalization type ('standard' or 'minmax', paper uses minmax)
        """
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.norm_type = norm_type
        
        self.feature_cols = ['lat', 'lon', 'sog', 'cog']
        self.scaler = StandardScaler()
        self.scaler_fitted = False
        
        # Min-max normalization parameters (paper uses this)
        self.feature_min = None
        self.feature_max = None
        
        # Target normalization parameters
        self.target_mean = None
        self.target_std = None
        self.target_min = None
        self.target_max = None
        self.max_remaining_hours = 720  # Filter out remaining time > 30 days
        
    def get_all_files(self) -> List[str]:
        """Get list of all AIS CSV files."""
        return sorted(glob.glob(os.path.join(self.data_dir, '*-ais.csv')))
        
    def load_all_data(self) -> pd.DataFrame:
        """Load all AIS CSV files and concatenate."""
        all_files = self.get_all_files()
        
        dfs = []
        for file in sorted(all_files):
            print(f"Loading {os.path.basename(file)}...")
            df = pd.read_csv(file)
            dfs.append(df)
            
        if not dfs:
            raise ValueError(f"No AIS files found in {self.data_dir}")
            
        combined = pd.concat(dfs, ignore_index=True)
        print(f"Total records loaded: {len(combined)}")
        return combined
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess AIS data.
        
        Steps:
        1. Parse timestamps
        2. Remove invalid data (null coordinates, invalid speeds)
        3. Calculate remaining time to ETA
        4. Sort by vessel and time
        """
        # Parse timestamps
        df['postime'] = pd.to_datetime(df['postime'])
        df['eta'] = pd.to_datetime(df['eta'], errors='coerce')
        
        # Remove rows with missing critical data
        df = df.dropna(subset=['lat', 'lon', 'sog', 'cog', 'postime', 'eta', 'mmsi'])
        
        # Filter valid speed and course
        df = df[(df['sog'] >= 0) & (df['sog'] <= 30)]  # Valid speed range
        df = df[(df['cog'] >= 0) & (df['cog'] <= 360)]  # Valid course range
        
        # Calculate remaining time (target variable)
        # Remove timezone info for subtraction
        df['eta_naive'] = df['eta'].dt.tz_localize(None)
        df['remaining_hours'] = (df['eta_naive'] - df['postime']).dt.total_seconds() / 3600
        
        # Filter positive remaining time (ship hasn't arrived yet)
        df = df[df['remaining_hours'] > 0]
        
        # Filter out unrealistic remaining times (> max_remaining_hours)
        before_filter = len(df)
        df = df[df['remaining_hours'] <= self.max_remaining_hours]
        filtered_out = before_filter - len(df)
        if filtered_out > 0:
            print(f"  Filtered out {filtered_out} records with remaining_hours > {self.max_remaining_hours}")
        
        # Sort by vessel and time
        df = df.sort_values(['mmsi', 'postime']).reset_index(drop=True)
        
        print(f"Records after preprocessing: {len(df)}")
        return df
    
    def extract_time_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract time features for temporal embedding.
        
        Returns array with columns: [minute, hour, day, weekday, month]
        Normalized to [-0.5, 0.5]
        """
        dt = df['postime']
        
        features = np.column_stack([
            dt.dt.minute.values / 60.0 - 0.5,      # [0, 59] -> [-0.5, 0.5]
            dt.dt.hour.values / 24.0 - 0.5,        # [0, 23] -> [-0.5, 0.5]
            (dt.dt.day.values - 1) / 31.0 - 0.5,   # [1, 31] -> [-0.5, 0.5]
            dt.dt.weekday.values / 7.0 - 0.5,      # [0, 6] -> [-0.5, 0.5]
            (dt.dt.month.values - 1) / 12.0 - 0.5  # [1, 12] -> [-0.5, 0.5]
        ])
        
        return features.astype(np.float32)
    
    def create_sequences_by_vessel(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create input/output sequences for each vessel trajectory.
        
        Returns:
            X: Input features (N, seq_len, num_features)
            X_mark: Input time features (N, seq_len, 5)
            y: Target remaining hours (N,)
            sailing_days: Days since start of voyage (N,)
        """
        total_len = self.seq_len + self.pred_len
        
        X_list = []
        X_mark_list = []
        y_list = []
        sailing_days_list = []
        
        # Extract features
        features = df[self.feature_cols].values
        time_features = self.extract_time_features(df)
        targets = df['remaining_hours'].values
        postimes = df['postime'].values
        mmsis = df['mmsi'].values
        
        # Group by vessel
        unique_mmsi = df['mmsi'].unique()
        print(f"Processing {len(unique_mmsi)} vessels...")
        
        for mmsi in unique_mmsi:
            mask = mmsis == mmsi
            vessel_features = features[mask]
            vessel_time = time_features[mask]
            vessel_targets = targets[mask]
            vessel_postimes = postimes[mask]
            
            n_points = len(vessel_features)
            if n_points < total_len:
                continue
            
            # Calculate sailing time from start
            start_time = vessel_postimes[0]
            
            # Create sliding window sequences
            for i in range(n_points - total_len + 1):
                X_list.append(vessel_features[i:i + self.seq_len])
                X_mark_list.append(vessel_time[i:i + self.seq_len])
                y_list.append(vessel_targets[i + self.seq_len - 1 + self.pred_len])
                
                # Calculate sailing days at prediction point
                current_time = vessel_postimes[i + self.seq_len - 1]
                sailing_days = (current_time - start_time) / np.timedelta64(1, 'D')
                sailing_days_list.append(sailing_days)
        
        X = np.array(X_list, dtype=np.float32)
        X_mark = np.array(X_mark_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        sailing_days = np.array(sailing_days_list, dtype=np.float32)
        
        print(f"Created {len(X)} sequences")
        return X, X_mark, y, sailing_days
    
    def normalize_features(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """Normalize features using specified method (standard or minmax)."""
        original_shape = X.shape
        X_flat = X.reshape(-1, X.shape[-1])
        
        if self.norm_type == 'minmax':
            # Min-max normalization as in the paper (range [0, 1])
            if fit and self.feature_min is None:
                self.feature_min = np.min(X_flat, axis=0)
                self.feature_max = np.max(X_flat, axis=0)
                # Avoid division by zero
                self.feature_max = np.where(self.feature_max == self.feature_min, 
                                           self.feature_min + 1, self.feature_max)
            
            X_normalized = (X_flat - self.feature_min) / (self.feature_max - self.feature_min)
        else:
            # Standard normalization
            if fit and not self.scaler_fitted:
                X_normalized = self.scaler.fit_transform(X_flat)
                self.scaler_fitted = True
            else:
                X_normalized = self.scaler.transform(X_flat)
            
        return X_normalized.reshape(original_shape).astype(np.float32)
    
    def normalize_target(self, y: np.ndarray, fit: bool = True) -> np.ndarray:
        """Normalize target variable using log transform + standardization."""
        # Use log(1 + y) to handle the skewed distribution
        y_log = np.log1p(y)
        
        if fit:
            self.target_mean = np.mean(y_log)
            self.target_std = np.std(y_log)
            if self.target_std == 0:
                self.target_std = 1.0
        
        y_normalized = (y_log - self.target_mean) / self.target_std
        return y_normalized.astype(np.float32)
    
    def inverse_normalize_target(self, y_normalized: np.ndarray) -> np.ndarray:
        """Inverse transform normalized target to original scale."""
        y_log = y_normalized * self.target_std + self.target_mean
        y = np.expm1(y_log)  # exp(y_log) - 1
        return y
    
    def partial_fit_scaler(self, X: np.ndarray):
        """Incrementally fit the scaler/normalizer with a batch of data."""
        X_flat = X.reshape(-1, X.shape[-1])
        
        if self.norm_type == 'minmax':
            # Update min-max values
            batch_min = np.min(X_flat, axis=0)
            batch_max = np.max(X_flat, axis=0)
            
            if self.feature_min is None:
                self.feature_min = batch_min
                self.feature_max = batch_max
            else:
                self.feature_min = np.minimum(self.feature_min, batch_min)
                self.feature_max = np.maximum(self.feature_max, batch_max)
        else:
            # Standard scaler
            if not self.scaler_fitted:
                self.scaler.fit(X_flat)
                self.scaler_fitted = True
            else:
                # Combine old and new statistics
                n_old = self.scaler.n_samples_seen_
                n_new = len(X_flat)
                n_total = n_old + n_new
                
                mean_old = self.scaler.mean_
                mean_new = np.mean(X_flat, axis=0)
                
                var_old = self.scaler.var_
                var_new = np.var(X_flat, axis=0)
                
                # Welford's online algorithm for combining means and variances
                self.scaler.mean_ = (n_old * mean_old + n_new * mean_new) / n_total
                self.scaler.var_ = (n_old * (var_old + (mean_old - self.scaler.mean_)**2) + 
                                   n_new * (var_new + (mean_new - self.scaler.mean_)**2)) / n_total
                self.scaler.scale_ = np.sqrt(self.scaler.var_)
                self.scaler.scale_[self.scaler.scale_ == 0] = 1.0
                self.scaler.n_samples_seen_ = n_total
            self.scaler.scale_ = np.sqrt(self.scaler.var_)
            self.scaler.scale_[self.scaler.scale_ == 0] = 1.0
            self.scaler.n_samples_seen_ = n_total
    
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Complete data preparation pipeline."""
        # Load and preprocess
        df = self.load_all_data()
        df = self.preprocess(df)
        
        # Create sequences
        X, X_mark, y, sailing_days = self.create_sequences_by_vessel(df)
        
        # Normalize features
        X = self.normalize_features(X, fit=True)
        
        return X, X_mark, y, sailing_days    
    def process_file_batch(self, file_path: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Process a single CSV file and return sequences.
        
        Returns None if file produces no valid sequences.
        """
        print(f"Processing {os.path.basename(file_path)}...")
        
        try:
            df = pd.read_csv(file_path)
            df = self.preprocess(df)
            
            if len(df) < self.seq_len + self.pred_len:
                print(f"  Skipping: not enough data after preprocessing")
                return None
            
            X, X_mark, y, sailing_days = self.create_sequences_by_vessel(df)
            
            if len(X) == 0:
                print(f"  Skipping: no valid sequences created")
                return None
            
            # Clear dataframe to free memory
            del df
            gc.collect()
            
            return X, X_mark, y, sailing_days
        except Exception as e:
            print(f"  Error processing file: {e}")
            return None
    
    def prepare_data_batched(self, max_files: Optional[int] = None, 
                             save_dir: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data by processing files in batches to avoid OOM.
        Uses memory-mapped files to avoid loading all data into RAM during concatenation.
        
        Args:
            max_files: Maximum number of files to process (None for all)
            save_dir: Directory to save intermediate results (required for large datasets)
            
        Returns:
            X, X_mark, y, sailing_days arrays
        """
        all_files = self.get_all_files()
        
        if max_files is not None:
            all_files = all_files[:max_files]
        
        print(f"Processing {len(all_files)} files in batches...")
        
        # First pass: fit scaler and count total sequences
        print("\n=== Pass 1: Fitting scaler and counting sequences ===")
        total_sequences = 0
        file_sequence_counts = []
        all_y = []  # Collect targets for normalization fitting
        
        for file_path in all_files:
            result = self.process_file_batch(file_path)
            if result is not None:
                X, _, y, _ = result
                self.partial_fit_scaler(X)
                all_y.append(y)
                n_seq = len(X)
                total_sequences += n_seq
                file_sequence_counts.append((file_path, n_seq))
                del result, X
                gc.collect()
            else:
                file_sequence_counts.append((file_path, 0))
        
        print(f"Total sequences to create: {total_sequences}")
        
        if total_sequences == 0:
            raise ValueError("No valid sequences found in any file")
        
        # Fit target normalization
        print("Fitting target normalization...")
        all_y_concat = np.concatenate(all_y)
        self.normalize_target(all_y_concat, fit=True)
        print(f"  Target stats: mean(log)={self.target_mean:.4f}, std(log)={self.target_std:.4f}")
        del all_y, all_y_concat
        gc.collect()
        
        # Setup save directory
        if save_dir is None:
            save_dir = os.path.join(self.data_dir, 'processed')
        os.makedirs(save_dir, exist_ok=True)
        
        # Create memory-mapped files for output
        print("\n=== Pass 2: Writing normalized sequences to disk ===")
        X_shape = (total_sequences, self.seq_len, len(self.feature_cols))
        X_mark_shape = (total_sequences, self.seq_len, 5)
        
        X_mmap = np.memmap(os.path.join(save_dir, 'X.dat'), dtype='float32', 
                          mode='w+', shape=X_shape)
        X_mark_mmap = np.memmap(os.path.join(save_dir, 'X_mark.dat'), dtype='float32',
                               mode='w+', shape=X_mark_shape)
        y_mmap = np.memmap(os.path.join(save_dir, 'y.dat'), dtype='float32',
                          mode='w+', shape=(total_sequences,))
        sailing_days_mmap = np.memmap(os.path.join(save_dir, 'sailing_days.dat'), dtype='float32',
                                     mode='w+', shape=(total_sequences,))
        
        # Second pass: write normalized data directly to memmap
        current_idx = 0
        for file_path, n_seq in file_sequence_counts:
            if n_seq == 0:
                continue
                
            print(f"Processing {os.path.basename(file_path)}... ({n_seq} sequences)")
            result = self.process_file_batch(file_path)
            
            if result is not None:
                X, X_mark, y, sailing_days = result
                X = self.normalize_features(X, fit=False)
                y = self.normalize_target(y, fit=False)  # Normalize target
                
                # Write directly to memmap
                end_idx = current_idx + len(X)
                X_mmap[current_idx:end_idx] = X
                X_mark_mmap[current_idx:end_idx] = X_mark
                y_mmap[current_idx:end_idx] = y
                sailing_days_mmap[current_idx:end_idx] = sailing_days
                
                current_idx = end_idx
                
                del result, X, X_mark, y, sailing_days
                gc.collect()
        
        # Flush memmap to disk
        X_mmap.flush()
        X_mark_mmap.flush()
        y_mmap.flush()
        sailing_days_mmap.flush()
        
        # Save metadata
        metadata = {
            'X_shape': X_shape,
            'X_mark_shape': X_mark_shape,
            'total_sequences': total_sequences,
            'seq_len': self.seq_len,
            'label_len': self.label_len,
            'pred_len': self.pred_len,
            'feature_cols': self.feature_cols,
            'target_mean': self.target_mean,
            'target_std': self.target_std,
            'max_remaining_hours': self.max_remaining_hours,
            'target_normalized': True,  # Flag to indicate targets are normalized
            'norm_type': self.norm_type,
            'feature_min': self.feature_min,
            'feature_max': self.feature_max,
        }
        with open(os.path.join(save_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"\nSaved processed data to {save_dir}")
        print(f"Total sequences: {total_sequences}")
        
        # Return as regular numpy arrays (load from memmap)
        # For very large datasets, use load_processed_data_mmap instead
        X = np.array(X_mmap)
        X_mark = np.array(X_mark_mmap)
        y = np.array(y_mmap)
        sailing_days = np.array(sailing_days_mmap)
        
        # Close memmaps
        del X_mmap, X_mark_mmap, y_mmap, sailing_days_mmap
        gc.collect()
        
        return X, X_mark, y, sailing_days
    
    def load_processed_data(self, save_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load previously processed and saved data."""
        # Try memmap format first
        if os.path.exists(os.path.join(save_dir, 'metadata.pkl')):
            return self.load_processed_data_mmap(save_dir)
        
        # Fall back to npy format
        X = np.load(os.path.join(save_dir, 'X.npy'))
        X_mark = np.load(os.path.join(save_dir, 'X_mark.npy'))
        y = np.load(os.path.join(save_dir, 'y.npy'))
        sailing_days = np.load(os.path.join(save_dir, 'sailing_days.npy'))
        
        with open(os.path.join(save_dir, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
            self.scaler_fitted = True
        
        print(f"Loaded {len(X)} sequences from {save_dir}")
        return X, X_mark, y, sailing_days
    
    def load_processed_data_mmap(self, save_dir: str, load_to_memory: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load previously processed data from memory-mapped files.
        
        Args:
            save_dir: Directory containing processed data
            load_to_memory: If True, load all data into RAM. If False, return memmaps.
        """
        with open(os.path.join(save_dir, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        with open(os.path.join(save_dir, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
            self.scaler_fitted = True
        
        # Restore target normalization parameters
        if 'target_mean' in metadata:
            self.target_mean = metadata['target_mean']
            self.target_std = metadata['target_std']
            print(f"  Target normalization: mean(log)={self.target_mean:.4f}, std(log)={self.target_std:.4f}")
        
        # Restore feature normalization parameters
        if 'norm_type' in metadata:
            self.norm_type = metadata['norm_type']
        if 'feature_min' in metadata:
            self.feature_min = metadata['feature_min']
            self.feature_max = metadata['feature_max']
        
        X_mmap = np.memmap(os.path.join(save_dir, 'X.dat'), dtype='float32',
                          mode='r', shape=metadata['X_shape'])
        X_mark_mmap = np.memmap(os.path.join(save_dir, 'X_mark.dat'), dtype='float32',
                               mode='r', shape=metadata['X_mark_shape'])
        y_mmap = np.memmap(os.path.join(save_dir, 'y.dat'), dtype='float32',
                          mode='r', shape=(metadata['total_sequences'],))
        sailing_days_mmap = np.memmap(os.path.join(save_dir, 'sailing_days.dat'), dtype='float32',
                                     mode='r', shape=(metadata['total_sequences'],))
        
        print(f"Loaded {metadata['total_sequences']} sequences from {save_dir}")
        
        if load_to_memory:
            X = np.array(X_mmap)
            X_mark = np.array(X_mark_mmap)
            y = np.array(y_mmap)
            sailing_days = np.array(sailing_days_mmap)
            del X_mmap, X_mark_mmap, y_mmap, sailing_days_mmap
            return X, X_mark, y, sailing_days
        
        return X_mmap, X_mark_mmap, y_mmap, sailing_days_mmap

class ETADataset(Dataset):
    """PyTorch Dataset for ETA prediction. Supports both numpy arrays and memmaps."""
    
    def __init__(self, X: np.ndarray, X_mark: np.ndarray, y: np.ndarray, 
                 sailing_days: np.ndarray, label_len: int = 24, pred_len: int = 1,
                 indices: Optional[np.ndarray] = None):
        """
        Args:
            X: Input features (N, seq_len, num_features) - can be memmap
            X_mark: Time features (N, seq_len, 5) - can be memmap
            y: Target values (N,) - can be memmap
            sailing_days: Sailing days for each sample (N,) - can be memmap
            label_len: Start token length
            pred_len: Prediction length
            indices: Optional subset indices to use
        """
        self.X = X
        self.X_mark = X_mark
        self.y = y
        self.sailing_days = sailing_days
        self.label_len = label_len
        self.pred_len = pred_len
        self.indices = indices if indices is not None else np.arange(len(X))
        
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get a single sample.
        
        Returns:
            x_enc: Encoder input (seq_len, features)
            x_mark_enc: Encoder time features (seq_len, 5)
            x_dec: Decoder input (label_len + pred_len, features)
            x_mark_dec: Decoder time features (label_len + pred_len, 5)
            y: Target remaining hours
            sailing_days: Days since voyage start
        """
        real_idx = self.indices[idx]
        
        # Load from memmap/array
        x = self.X[real_idx]
        x_mark = self.X_mark[real_idx]
        y_val = self.y[real_idx]
        sailing_day = self.sailing_days[real_idx]
        
        # Convert to tensors
        x_enc = torch.FloatTensor(x)
        x_mark_enc = torch.FloatTensor(x_mark)
        
        # Decoder input: [start_token (last label_len), zeros (pred_len)]
        start_token = torch.FloatTensor(x[-self.label_len:])
        zeros = torch.zeros(self.pred_len, x.shape[-1])
        x_dec = torch.cat([start_token, zeros], dim=0)
        
        # Decoder time features
        start_time_mark = torch.FloatTensor(x_mark[-self.label_len:])
        pred_time_mark = torch.FloatTensor(x_mark[-1:]).expand(self.pred_len, -1)
        x_mark_dec = torch.cat([start_time_mark, pred_time_mark], dim=0)
        
        return (x_enc, x_mark_enc, x_dec, x_mark_dec, 
                torch.tensor(y_val, dtype=torch.float32), 
                torch.tensor(sailing_day, dtype=torch.float32))


def create_data_loaders(X: np.ndarray, X_mark: np.ndarray, y: np.ndarray,
                        sailing_days: np.ndarray, label_len: int = 24, pred_len: int = 1,
                        batch_size: int = 32, train_ratio: float = 0.7,
                        val_ratio: float = 0.15) -> Tuple[DataLoader, DataLoader, DataLoader,
                                                           np.ndarray, np.ndarray, np.ndarray]:
    """
    Create train/val/test data loaders. Works with both numpy arrays and memmaps.
    
    Returns:
        train_loader, val_loader, test_loader,
        test_sailing_days, test_y_true, test_indices
    """
    n_samples = len(X)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    # Shuffle indices
    indices = np.random.permutation(n_samples)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    # Create datasets using index subsetting (works with memmaps)
    train_dataset = ETADataset(X, X_mark, y, sailing_days, label_len, pred_len, indices=train_idx)
    val_dataset = ETADataset(X, X_mark, y, sailing_days, label_len, pred_len, indices=val_idx)
    test_dataset = ETADataset(X, X_mark, y, sailing_days, label_len, pred_len, indices=test_idx)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Get test data for evaluation
    test_sailing_days = sailing_days[test_idx] if isinstance(sailing_days, np.ndarray) else np.array([sailing_days[i] for i in test_idx])
    test_y_true = y[test_idx] if isinstance(y, np.ndarray) else np.array([y[i] for i in test_idx])
    
    return (train_loader, val_loader, test_loader,
            test_sailing_days, test_y_true, test_idx)


if __name__ == "__main__":
    # Test data loading
    processor = ETADataProcessor(
        data_dir="/home/wsy/ETA/data",
        seq_len=48,
        label_len=24,
        pred_len=1
    )
    
    X, X_mark, y, sailing_days = processor.prepare_data()
    print(f"X shape: {X.shape}")
    print(f"X_mark shape: {X_mark.shape}")
    print(f"y shape: {y.shape}")
    print(f"sailing_days shape: {sailing_days.shape}")
    print(f"y range: [{y.min():.2f}, {y.max():.2f}] hours")
    print(f"sailing_days range: [{sailing_days.min():.2f}, {sailing_days.max():.2f}] days")
