"""Generate figures for the ETA paper."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def draw_architecture():
    """Generate system architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Colors
    c_data = '#E3F2FD'      # light blue
    c_informer = '#FFF3E0'  # light orange  
    c_port = '#E8F5E9'      # light green
    c_output = '#FCE4EC'    # light pink
    c_embed = '#F3E5F5'     # light purple
    c_enc = '#FFE0B2'       # encoder orange
    c_dec = '#FFCCBC'       # decoder salmon
    c_border = '#37474F'    # dark gray
    
    def box(x, y, w, h, text, color, fontsize=8, bold=False):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor=c_border, linewidth=1.2)
        ax.add_patch(rect)
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
                fontsize=fontsize, fontweight=weight, wrap=True)
    
    def arrow(x1, y1, x2, y2, style='->', color='#455A64'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=1.5))
    
    # ============ Data Input (left) ============
    box(0.1, 4.2, 1.6, 1.3, 'AIS Data\n(lat, lon, SOG,\nCOG, dist, Δθ)', c_data, 7.5, True)
    box(0.1, 2.6, 1.6, 1.3, 'Weather Data\n(temp, wind,\npressure, vis)', c_data, 7.5, True)
    
    # ============ Embedding ============
    box(2.2, 3.0, 1.4, 2.0, 'Input\nEmbedding\n\nConv1d +\nPE + TE', c_embed, 7.5)
    
    arrow(1.7, 4.85, 2.2, 4.3)
    arrow(1.7, 3.25, 2.2, 3.7)
    
    # ============ Informer (center) ============
    # Encoder block
    box(4.0, 3.8, 1.8, 1.5, 'Encoder (×2)\n\nProbSparse\nSelf-Attention\n+ Distilling', c_enc, 7.5, True)
    
    # Decoder block
    box(6.2, 3.8, 1.8, 1.5, 'Decoder (×1)\n\nMasked Attn\n+ Cross-Attn', c_dec, 7.5, True)
    
    # FC output
    box(8.4, 4.05, 1.3, 1.0, 'Linear\nProjection\n→ ŷ_sail', c_output, 7.5)
    
    arrow(3.6, 4.55, 4.0, 4.55)
    arrow(5.8, 4.55, 6.2, 4.55)
    arrow(8.0, 4.55, 8.4, 4.55)
    
    # Cross-attention arrow from encoder to decoder (curved)
    ax.annotate('', xy=(7.1, 3.8), xytext=(4.9, 3.8),
                arrowprops=dict(arrowstyle='->', color='#D84315', lw=1.5,
                               connectionstyle='arc3,rad=-0.3'))
    ax.text(6.0, 3.25, 'cross-attn', fontsize=6, ha='center', color='#D84315', style='italic')
    
    # ============ Port Model (bottom) ============
    box(4.0, 1.0, 1.8, 1.5, 'Port MLP\n\n64 → 32 → 1\nRegion + Time', c_port, 7.5, True)
    
    box(0.1, 1.0, 1.6, 1.3, 'Port Features\n(region, arrival\ntime, coords)', c_data, 7.5, True)
    
    arrow(1.7, 1.65, 4.0, 1.75)
    
    box(6.2, 1.0, 1.8, 1.0, 'ŷ_port', c_output, 9)
    arrow(5.8, 1.75, 6.2, 1.5)
    
    # ============ Final summation ============
    # Sum circle
    circle = plt.Circle((9.2, 2.75), 0.3, fill=True, facecolor='#FFEB3B', 
                         edgecolor=c_border, linewidth=1.5)
    ax.add_patch(circle)
    ax.text(9.2, 2.75, '∑', ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Arrows to sum
    arrow(9.05, 4.05, 9.15, 3.05)  # from sail
    arrow(8.0, 1.5, 9.0, 2.5)  # from port
    
    # Final output
    ax.text(9.2, 2.15, 'ETA_total', ha='center', va='center', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4', edgecolor=c_border))
    
    # ============ Labels ============
    ax.text(5.0, 5.65, 'Sailing Time Predictor (Informer)', ha='center', fontsize=10, 
            fontweight='bold', color='#BF360C')
    ax.text(4.9, 0.35, 'Port Dwell Time Predictor (MLP)', ha='center', fontsize=10, 
            fontweight='bold', color='#1B5E20')
    
    # Dashed separator
    ax.axhline(y=2.7, xmin=0.02, xmax=0.83, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('ETA-paper/architecture.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('ETA-paper/architecture.png', dpi=200, bbox_inches='tight')
    print("Saved architecture.pdf/png")


def draw_training_curve():
    """Generate training/validation loss curves."""
    epochs = [1, 2, 3, 4, 5]
    train_loss = [0.0268, 0.0124, 0.0078, 0.0052, 0.0039]
    val_loss = [0.0345, 0.0194, 0.0247, 0.0212, 0.0235]
    
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3))
    
    ax.plot(epochs, train_loss, 'o-', color='#1565C0', linewidth=2, markersize=6, label='Training Loss')
    ax.plot(epochs, val_loss, 's-', color='#E65100', linewidth=2, markersize=6, label='Validation Loss')
    
    # Mark best epoch
    best_epoch = 2
    ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax.annotate('Best (Epoch 2)', xy=(best_epoch, val_loss[1]), xytext=(3.2, 0.032),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.2),
                fontsize=8, color='green')
    
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Loss (Huber)', fontsize=10)
    ax.set_xticks(epochs)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 5.5)
    
    plt.tight_layout()
    plt.savefig('ETA-paper/training_curve.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('ETA-paper/training_curve.png', dpi=200, bbox_inches='tight')
    print("Saved training_curve.pdf/png")


def draw_baseline_comparison():
    """Generate baseline MAE comparison bar chart with all models grouped by family."""
    # Results grouped by family (excluding Ridge for main chart)
    models = ['MLP', 'LSTM', 'GRU', '1D-CNN', 'TCN', 'XGBoost', 
              'Transformer', 'Conv-Trans.', 'Informer',
              'MSTGN-MLP', 'StatMLP', 'MSTGN-Late']
    maes = [15.73, 15.79, 16.03, 16.08, 17.19, 15.14, 
            18.12, 16.60, 16.20,
            15.40, 15.41, 15.56]
    
    # Color by family
    family_colors = {
        'Feedforward': '#90CAF9',   # blue
        'Recurrent': '#A5D6A7',     # green  
        'Convolutional': '#FFCC80', # orange
        'Tree': '#EF9A9A',          # red
        'Attention': '#CE93D8',     # purple
        'Graph': '#80CBC4',         # teal
    }
    families = ['Feedforward', 'Recurrent', 'Recurrent', 'Convolutional', 'Convolutional',
                'Tree', 'Attention', 'Attention', 'Attention',
                'Graph', 'Graph', 'Graph']
    colors = [family_colors[f] for f in families]
    
    # Sort by MAE for visual clarity
    sorted_indices = np.argsort(maes)
    models_sorted = [models[i] for i in sorted_indices]
    maes_sorted = [maes[i] for i in sorted_indices]
    colors_sorted = [colors[i] for i in sorted_indices]
    families_sorted = [families[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    
    y_pos = range(len(models_sorted))
    bars = ax.barh(y_pos, maes_sorted, color=colors_sorted, edgecolor='#424242', linewidth=0.5, height=0.7)
    
    ax.set_xlim(14, 19)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models_sorted, fontsize=9)
    ax.set_xlabel('MAE (hours)', fontsize=10)
    
    for i, v in enumerate(maes_sorted):
        ax.text(v + 0.05, i, f'{v:.2f}h', va='center', fontsize=8)
    
    # Best model line
    ax.axvline(x=15.14, color='#C62828', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(15.20, len(models_sorted) - 0.3, 'Best (XGBoost)', fontsize=7, color='#C62828', alpha=0.7)
    
    # Legend for families
    legend_handles = [mpatches.Patch(facecolor=c, edgecolor='#424242', label=f) 
                      for f, c in family_colors.items()]
    ax.legend(handles=legend_handles, fontsize=7, loc='lower right', 
              title='Architecture Family', title_fontsize=8)
    
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_title('ETA Prediction: MAE Comparison Across Architectures\n(Ridge Regression omitted: MAE = 58.30h)', 
                 fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('ETA-paper/baseline_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('ETA-paper/baseline_comparison.png', dpi=200, bbox_inches='tight')
    print("Saved baseline_comparison.pdf/png")


if __name__ == '__main__':
    draw_architecture()
    draw_training_curve()
    draw_baseline_comparison()
    draw_training_curve()
    draw_baseline_comparison()
    print("\nAll figures generated.")
