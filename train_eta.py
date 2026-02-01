"""
Training and Evaluation script for ETA prediction using Informer-TP.

This script:
1. Loads and preprocesses AIS data
2. Trains the Informer-TP model
3. Evaluates performance on test set
4. Plots prediction accuracy vs sailing days
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.informer.model import Informer
from src.data_processor import ETADataProcessor, create_data_loaders


class ETATrainer:
    """Trainer for Informer ETA prediction model."""
    
    def __init__(self, model: nn.Module, device: torch.device,
                 learning_rate: float = 1e-4, weight_decay: float = 1e-5):
        self.model = model.to(device)
        self.device = device
        # Use Huber Loss (Smooth L1) - less sensitive to outliers than MSE
        self.criterion = nn.HuberLoss(delta=1.0)
        self.optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3)
        
    def train_epoch(self, train_loader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            x_enc, x_mark_enc, x_dec, x_mark_dec, y, _ = batch
            
            # Move to device
            x_enc = x_enc.to(self.device)
            x_mark_enc = x_mark_enc.to(self.device)
            x_dec = x_dec.to(self.device)
            x_mark_dec = x_mark_dec.to(self.device)
            y = y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            # Loss on prediction (squeeze output to match target shape)
            output = output.squeeze(-1).squeeze(-1)  # (batch,)
            loss = self.criterion(output, y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def validate(self, val_loader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x_enc, x_mark_enc, x_dec, x_mark_dec, y, _ = batch
                
                x_enc = x_enc.to(self.device)
                x_mark_enc = x_mark_enc.to(self.device)
                x_dec = x_dec.to(self.device)
                x_mark_dec = x_mark_dec.to(self.device)
                y = y.to(self.device)
                
                output = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                output = output.squeeze(-1).squeeze(-1)
                loss = self.criterion(output, y)
                
                total_loss += loss.item()
                
        return total_loss / len(val_loader)
    
    def predict(self, test_loader) -> tuple:
        """Make predictions on test set."""
        self.model.eval()
        predictions = []
        targets = []
        sailing_days_list = []
        
        with torch.no_grad():
            for batch in test_loader:
                x_enc, x_mark_enc, x_dec, x_mark_dec, y, sailing_days = batch
                
                x_enc = x_enc.to(self.device)
                x_mark_enc = x_mark_enc.to(self.device)
                x_dec = x_dec.to(self.device)
                x_mark_dec = x_mark_dec.to(self.device)
                
                output = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                output = output.squeeze(-1).squeeze(-1)
                
                predictions.extend(output.cpu().numpy())
                targets.extend(y.numpy())
                sailing_days_list.extend(sailing_days.numpy())
                
        return (np.array(predictions), np.array(targets), np.array(sailing_days_list))
    
    def save_model(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        
    def load_model(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])


def calculate_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    """Calculate evaluation metrics."""
    # MSE
    mse = np.mean((y_pred - y_true) ** 2)
    
    # MAE
    mae = np.mean(np.abs(y_pred - y_true))
    
    # RMSE
    rmse = np.sqrt(mse)
    
    # MAPE (avoid division by zero, only for y_true > 24 hours)
    mask = y_true > 24  # Only calculate MAPE for remaining time > 1 day
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100
    else:
        mape = float('nan')
    
    # MAE in days for readability
    mae_days = mae / 24
    
    return {
        'MSE': mse,
        'MAE_hours': mae,
        'MAE_days': mae_days,
        'RMSE': rmse,
        'MAPE (y>24h)': mape
    }


def plot_accuracy_vs_sailing_days(y_pred: np.ndarray, y_true: np.ndarray, 
                                   sailing_days: np.ndarray, save_path: str = None):
    """
    Plot prediction accuracy vs sailing days.
    
    Groups predictions by sailing days and calculates MAE for each group.
    """
    plt.figure(figsize=(12, 8))
    
    # Calculate absolute errors
    errors = np.abs(y_pred - y_true)
    
    # Create bins for sailing days
    max_days = min(sailing_days.max(), 30)  # Cap at 30 days
    bins = np.arange(0, max_days + 1, 1)  # 1-day bins
    bin_indices = np.digitize(sailing_days, bins) - 1
    
    # Calculate metrics for each bin
    bin_maes = []
    bin_centers = []
    bin_counts = []
    
    for i in range(len(bins) - 1):
        mask = bin_indices == i
        if np.sum(mask) > 10:  # Minimum samples per bin
            bin_mae = np.mean(errors[mask])
            bin_maes.append(bin_mae)
            bin_centers.append((bins[i] + bins[i+1]) / 2)
            bin_counts.append(np.sum(mask))
    
    # Plot 1: MAE vs Sailing Days
    plt.subplot(2, 2, 1)
    plt.bar(bin_centers, bin_maes, width=0.8, color='steelblue', alpha=0.7)
    plt.xlabel('Sailing Days', fontsize=12)
    plt.ylabel('Mean Absolute Error (hours)', fontsize=12)
    plt.title('Prediction Error vs Sailing Days', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: MAPE vs Sailing Days
    plt.subplot(2, 2, 2)
    bin_mapes = []
    for i in range(len(bins) - 1):
        mask = (bin_indices == i) & (y_true > 0)
        if np.sum(mask) > 10:
            bin_mape = np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100
            bin_mapes.append(bin_mape)
    
    if bin_mapes:
        plt.bar(bin_centers[:len(bin_mapes)], bin_mapes, width=0.8, color='coral', alpha=0.7)
        plt.xlabel('Sailing Days', fontsize=12)
        plt.ylabel('MAPE (%)', fontsize=12)
        plt.title('Percentage Error vs Sailing Days', fontsize=14)
        plt.grid(True, alpha=0.3)
    
    # Plot 3: Scatter plot of predictions vs actual
    plt.subplot(2, 2, 3)
    sample_idx = np.random.choice(len(y_pred), min(1000, len(y_pred)), replace=False)
    plt.scatter(y_true[sample_idx], y_pred[sample_idx], alpha=0.3, s=10)
    max_val = max(y_true[sample_idx].max(), y_pred[sample_idx].max())
    plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual Remaining Time (hours)', fontsize=12)
    plt.ylabel('Predicted Remaining Time (hours)', fontsize=12)
    plt.title('Predicted vs Actual ETA', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Sample count per bin
    plt.subplot(2, 2, 4)
    plt.bar(bin_centers, bin_counts, width=0.8, color='green', alpha=0.7)
    plt.xlabel('Sailing Days', fontsize=12)
    plt.ylabel('Sample Count', fontsize=12)
    plt.title('Data Distribution by Sailing Days', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.close()  # Close figure instead of show
    
    return bin_centers, bin_maes


def main():
    parser = argparse.ArgumentParser(description='Train Informer-TP for ETA Prediction')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--seq_len', type=int, default=48, help='Input sequence length')
    parser.add_argument('--label_len', type=int, default=24, help='Start token length')
    parser.add_argument('--pred_len', type=int, default=1, help='Prediction length')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--e_layers', type=int, default=2, help='Encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='Decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='Feedforward dimension')
    parser.add_argument('--dropout', type=float, default=0.05, help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-6, help='Learning rate (paper uses 5e-6)')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--max_files', type=int, default=None, help='Maximum files to process (for testing)')
    parser.add_argument('--use_cache', action='store_true', help='Use cached preprocessed data')
    parser.add_argument('--cache_dir', type=str, default='./output/cache', help='Cache directory for preprocessed data')
    parser.add_argument('--norm_type', type=str, default='minmax', choices=['standard', 'minmax'], help='Normalization type (paper uses minmax)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    print("\n=== Loading Data ===")
    processor = ETADataProcessor(
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        norm_type=args.norm_type
    )
    
    # Check for cached data or process in batches
    cache_exists = (
        os.path.exists(os.path.join(args.cache_dir, 'metadata.pkl')) or  # memmap format
        os.path.exists(os.path.join(args.cache_dir, 'X.npy'))  # npy format
    )
    
    if args.use_cache and cache_exists:
        print("Loading cached data...")
        X, X_mark, y, sailing_days = processor.load_processed_data(args.cache_dir)
    else:
        print("Processing data in batches (memory-efficient mode)...")
        X, X_mark, y, sailing_days = processor.prepare_data_batched(
            max_files=args.max_files,
            save_dir=args.cache_dir
        )
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Create data loaders
    print("\n=== Creating Data Loaders ===")
    train_loader, val_loader, test_loader, test_sailing_days, test_y, test_idx = create_data_loaders(
        X, X_mark, y, sailing_days,
        label_len=args.label_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Create model
    print("\n=== Creating Informer Model ===")
    model = Informer(
        enc_in=4,  # lat, lon, sog, cog
        dec_in=4,
        c_out=1,   # ETA (remaining time)
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_layers=args.d_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        attn='prob',
        embed='timeF',
        activation='gelu',
        output_attention=False,
        distil=True,
        device=device
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = ETATrainer(model, device, learning_rate=args.lr)
    
    # Training loop
    print("\n=== Training ===")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Update learning rate
        trainer.scheduler.step(val_loss)
        current_lr = trainer.optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{args.epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, LR={current_lr:.2e}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_model(os.path.join(args.output_dir, 'best_model.pth'))
            print("  -> Best model saved!")
    
    # Load best model for evaluation
    trainer.load_model(os.path.join(args.output_dir, 'best_model.pth'))
    
    # Evaluate on test set
    print("\n=== Evaluation ===")
    y_pred_norm, y_true_norm, pred_sailing_days = trainer.predict(test_loader)
    
    # Inverse transform predictions and targets to original scale
    y_pred = processor.inverse_normalize_target(y_pred_norm)
    y_true = processor.inverse_normalize_target(y_true_norm)
    
    print(f"Predictions range: [{y_pred.min():.2f}, {y_pred.max():.2f}] hours")
    print(f"Actual range: [{y_true.min():.2f}, {y_true.max():.2f}] hours")
    
    # Calculate metrics (in original scale - hours)
    metrics = calculate_metrics(y_pred, y_true)
    print("\nTest Metrics (hours):")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Save metrics
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Training completed: {datetime.now().isoformat()}\n\n")
        f.write("Model Configuration:\n")
        for key, value in vars(args).items():
            f.write(f"  {key}: {value}\n")
        f.write("\nTest Metrics:\n")
        for name, value in metrics.items():
            f.write(f"  {name}: {value:.4f}\n")
    
    # Plot training curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.yscale('log')
    plt.title('Training Loss (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_curves.png'), dpi=150)
    plt.close()  # Close figure instead of show
    
    # Plot accuracy vs sailing days
    print("\n=== Plotting Results ===")
    plot_accuracy_vs_sailing_days(
        y_pred, y_true, pred_sailing_days,
        save_path=os.path.join(args.output_dir, 'accuracy_vs_sailing_days.png')
    )
    
    print(f"\nTraining complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
