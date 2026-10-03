import sys
import numpy as np
from pathlib import Path
import lightgbm as lgb
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from torch.utils.data import random_split
import warnings

# Ensure the project root is in the Python path to allow absolute imports
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.training.pe_dataset import PEGraphDataset
from src.utils.logger import ExperimentLogger
from src.utils.config import CONFIG

# Extract LightGBM specific parameters from config
lgbm_config = CONFIG["lgbm"]
warnings.filterwarnings("ignore")

def custom_eval_metrics(y_true, y_pred):
    # Convert LightGBM's continuous probabilities into binary 0/1 predictions
    y_bin = (y_pred > 0.5).astype(int)
    
    return [
        ('f1', f1_score(y_true, y_bin), True),
        ('accuracy', accuracy_score(y_true, y_bin), True),
        ('precision', precision_score(y_true, y_bin, zero_division=0), True),
        ('recall', recall_score(y_true, y_bin, zero_division=0), True)
    ]

def convert_graphs_to_tabular(dataset, desc="Converting Graphs"):
    """
    LightGBM requires 2D tabular data (Samples x Features). 
    This function iterates through PyTorch Geometric graphs and mean-pools 
    their node features to create a 1D feature vector for each graph.
    """
    X_list, y_list = [], []
    for data in tqdm(dataset, desc=desc):
        # Mean pooling across the nodes (dim=0) to collapse the graph into a flat vector
        graph_vec = data.x.mean(dim=0).numpy()
        X_list.append(graph_vec)
        
        # Ensure label is a 1D scalar
        label = data.y.item() if data.y.numel() == 1 else data.y[0].item()
        y_list.append(label)
        
    return np.array(X_list), np.array(y_list)

def train_baseline_lgbm():
    # 1. Initialize Logger
    logger = ExperimentLogger(experiment_name="lgbm_training", config=CONFIG)

    # 2. Load Data from new monthly split
    graphs_dir = project_root / CONFIG["graphs_path"]
    
    print("\n[Data] Loading January 2018 dataset (Training set)...")
    jan_dataset = PEGraphDataset(graphs_dir / "train")
    
    # Strictly internal 80/20 split on January data (same as GNN/GAN)
    train_size = int(0.8 * len(jan_dataset))
    val_size = len(jan_dataset) - train_size
    train_data, val_data = random_split(jan_dataset, [train_size, val_size])

    print("\n[Data] Converting Graph data to Tabular format for LightGBM...")
    X_train, y_train = convert_graphs_to_tabular(train_data, desc="Train Split")
    X_val, y_val = convert_graphs_to_tabular(val_data, desc="Val Split")

    # 3. Initialize Model
    model = lgb.LGBMClassifier(
        n_estimators=lgbm_config["n_estimators"],
        learning_rate=lgbm_config["learning_rate"],
        num_leaves=lgbm_config["num_leaves"],
        objective=lgbm_config.get("objective", "binary"),
        random_state=CONFIG["random_seed"],
        n_jobs=-1,
        verbose=-1,
    )

    # 4. Training Phase
    print(f"\n[Lgbm] Starting LightGBM training for {lgbm_config['n_estimators']} trees...")
    
    pbar = tqdm(total=lgbm_config["n_estimators"], desc="Building Trees", file=sys.stdout, dynamic_ncols=True)
    
    def tqdm_callback(env):
        pbar.update(1)
        
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)], 
        eval_names=['train', 'val'],
        eval_metric=["binary_logloss", "auc", custom_eval_metrics],         
        callbacks=[
            tqdm_callback, 
            lgb.log_evaluation(period=0)
        ]
    )
    pbar.close()
        
    # 5. Extract and Log Per-Iteration Metrics
    print("\n[Lgbm] Extracting per-iteration metrics for graphs...")
    results = model.evals_result_
    
    num_trees = len(results['train']['binary_logloss'])
    target_points = CONFIG["mlp"]["num_epochs"]
    step_size = max(1, num_trees // target_points)

    for step in range(1, target_points + 1):
        i = min((step * step_size) - 1, num_trees - 1)
        
        epoch_metrics = {
            "train_loss": results['train']['binary_logloss'][i],
            "test_loss": results['val']['binary_logloss'][i], # mapped to val
            "auc": results['val']['auc'][i],
            "f1": results['val']['f1'][i],
            "accuracy": results['val']['accuracy'][i],
            "precision": results['val']['precision'][i],
            "recall": results['val']['recall'][i]
        }
        logger.log_epoch(epoch=step, metrics=epoch_metrics)  

    # 6. Calculate Final Validation Metrics
    print("\n[Lgbm] Evaluating final validation performance...")
    preds_proba = model.predict_proba(X_val)[:, 1]
    preds_binary = model.predict(X_val)
    
    final_auc = roc_auc_score(y_val, preds_proba)
    final_f1 = f1_score(y_val, preds_binary)
    
    print(f"Validation Summary - AUC: {final_auc:.4f} | F1: {final_f1:.4f}")
        
    # 7. Save model
    model_path = logger.run_dir / "model_lgbm.txt"
    model.booster_.save_model(str(model_path))
    
    print(f"\n[Success] LightGBM Training complete!")
    print(f"  -> Model weights saved in: {model_path}")

if __name__ == "__main__":
    train_baseline_lgbm()