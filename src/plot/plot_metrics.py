import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys

# Ensure the project root is in the system path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.config import CONFIG

def get_latest_experiment_csv(experiments_dir: Path, prefix: str) -> Path:
    """Automatically finds the latest metrics.csv file for a given model prefix."""
    print(f"[DEBUG] Target Directory: {experiments_dir.resolve()}")
    search_pattern = f"{prefix}/*/metrics.csv"
    print(f"[DEBUG] Search Pattern: {search_pattern}")
    paths = list(experiments_dir.glob(search_pattern))
    print(f"[DEBUG] Found {len(paths)} files: {paths}")

    if not paths:
        return None  # Return None instead of crashing, useful if a model isn't trained yet
    # Return the most recently modified file
    return max(paths, key=os.path.getmtime)

def plot_comparisons(models_to_compare: list):
    print(f"Generating plots for: {', '.join(models_to_compare)}...")
    experiments_dir = project_root / CONFIG["experiments_path"]
    
    plt.style.use('default')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Use a default color palette to differentiate models automatically
    colors = plt.cm.tab10.colors 

    for i, model_prefix in enumerate(models_to_compare):
        csv_path = get_latest_experiment_csv(experiments_dir, model_prefix)
        
        if not csv_path:
            print(f"[Warning] No data found for '{model_prefix}', skipping.")
            continue
            
        data = pd.read_csv(csv_path)
        color = colors[i % len(colors)]

        # 1. Plot Loss
        if 'train_loss' in data.columns:
            axes[0].plot(data['epoch'], data['train_loss'], color=color, marker='o', 
                            markersize=4, label=f'{model_prefix} Train Loss')
        if 'test_loss' in data.columns:
            axes[0].plot(data['epoch'], data['test_loss'], color=color, marker='s', 
                            linestyle='--', markersize=4, label=f'{model_prefix} Test Loss')
        
        # 2. Plot AUC
        if 'auc' in data.columns:
            axes[1].plot(data['epoch'], data['auc'], color=color, marker='^', 
                            markersize=4, label=f'{model_prefix} AUC')

    # Setup Plot 1: Loss
    axes[0].set_title('Train vs Test Loss', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=10)
    axes[0].set_ylabel('Loss', fontsize=10)
    axes[0].legend(loc='best')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Setup Plot 2: AUC
    axes[1].set_title('Test AUC Comparison', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=10)
    axes[1].set_ylabel('AUC Score', fontsize=10)
    axes[1].set_ylim(0.5, 1.05)
    axes[1].legend(loc='lower right')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # Save the plot
    output_path = experiments_dir / 'model_comparison.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot successfully saved to:\n{output_path}")
    
    plt.show()

# Allows testing the script directly from the terminal
if __name__ == "__main__":
    plot_comparisons(["mlp_baseline", "lgbm_baseline"])