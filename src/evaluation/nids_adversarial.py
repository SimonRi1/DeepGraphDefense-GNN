import sys
import os
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
import joblib

# Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.gnn import MFGraph
from src.training.pe_dataset import PEGraphDataset
from src.utils.config import CONFIG


def compute_all_metrics(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    auc = np.nan
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = np.nan

    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "FPR": fpr,
        "ROC_AUC": auc
    }


def load_nids_gnn(device):
    """Loads the trained NIDS MFGraph from the latest timestamped folder."""
    gnn_config = CONFIG.get("gnn", {"hidden_dim": 64, "k": 10, "dropout_rate": 0.5})
    
    # Locate the experiments folder
    nids_dir = project_root / CONFIG.get("experiments_path", "experiments") / "nids_gnn"
    
    if not nids_dir.exists():
        raise FileNotFoundError(f"NIDS experiments directory not found at {nids_dir}")
        
    # Find all subdirectories and sort them descending (latest timestamp first)
    timestamp_folders = sorted([d for d in nids_dir.iterdir() if d.is_dir()], reverse=True)
    if not timestamp_folders:
        raise FileNotFoundError(f"No timestamped NIDS GNN folders found in {nids_dir}")
        
    # Select the most recent folder
    latest_run = timestamp_folders[0]
    
    # Search for checkpoints only inside the latest timestamped folder
    checkpoints = list(latest_run.rglob("*.pth")) + list(latest_run.rglob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No NIDS GNN checkpoint found in {latest_run}")
        
    nids_weights = max(checkpoints, key=lambda f: f.stat().st_mtime)
    print(f"[NIDS Adversarial] Loaded GNN weights from: {latest_run.name}/{nids_weights.name}")
    
    model = MFGraph(
        input_dim=9,
        hidden_dim=gnn_config.get("hidden_dim", 64),
        k=gnn_config.get("k", 10),
        dropout_rate=gnn_config.get("dropout_rate", 0.5)
    ).to(device)
    
    state_dict = torch.load(nids_weights, map_location=device)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    return model


def graphs_to_tabular(loader):
    """Collapses NIDS flow graphs into 1D tabular vectors using global mean pooling."""
    X, y = [], []
    for batch in loader:
        pooled = global_mean_pool(batch.x, batch.batch)
        X.append(pooled.cpu().numpy())
        
        if hasattr(batch, 'y') and batch.y is not None:
            labels = batch.y.cpu().numpy()
            if len(labels) == pooled.size(0):
                y.extend(labels)
            else:
                y.extend([labels[0]] * pooled.size(0))
        else:
            y.extend([0] * pooled.size(0))
            
    return np.vstack(X), np.array(y, dtype=np.int32)


def evaluate_nids_adversarial():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n========================================================")
    print(f"   NIDS ADVERSARIAL STRESS TEST (NETWORK DOMAIN)")
    print(f"========================================================")
    print(f"[Info] Running on device: {device}")
    
    # 1. Load NIDS Test Data
    nids_test_path = project_root / "data" / "graphs" / "nids_graphs"
    if not nids_test_path.exists():
        raise FileNotFoundError(f"NIDS graph data not found at {nids_test_path}")
        
    dataset = PEGraphDataset(str(nids_test_path))
    print(f"[Info] Total NIDS graphs loaded: {len(dataset)}")
    
    # Use 20% test split (matching training split seed 42)
    test_size = int(0.2 * len(dataset))
    train_size = len(dataset) - test_size
    generator = torch.Generator().manual_seed(42)
    _, test_data = torch.utils.data.random_split(dataset, [train_size, test_size], generator=generator)
    
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
    X_raw_test, y_true = graphs_to_tabular(test_loader)
    
    # 2. Load Models
    gnn_model = load_nids_gnn(device)
    
    baseline_dir = project_root / "experiments" / "NIDSbaselines_training"
    model_names = ["Random Forest", "Decision Tree ID3", "AdaBoost", "MLP"]
    baselines = {}
    
    if baseline_dir.exists():
        for name in model_names:
            filename = f"{name.replace(' ', '_').replace('(', '').replace(')', '')}.joblib"
            model_path = baseline_dir / filename
            if model_path.exists():
                baselines[name] = joblib.load(model_path)
                
    if not baselines:
        raise FileNotFoundError(f"No pre-trained NIDS baseline models found in {baseline_dir}. Run baseline training first.")

    # 3. Define Attacks Configuration
    attacks = {
        "Packet_Size": {"feat_idx": 2, "scales": [1.0, 0.8, 0.5, 0.1, 2.0, 5.0]},
        "Inter-Arrival_Time": {"feat_idx": 4, "scales": [1.0, 2.0, 5.0, 10.0]}
    }
    
    out_dir = project_root / "experiments" / "nids_adversarial"
    out_dir.mkdir(parents=True, exist_ok=True)

    for attack_name, config in attacks.items():
        print(f"\n--- Testing Adversarial Robustness: {attack_name.replace('_', ' ')} ---")
        feat_idx = config["feat_idx"]
        scales = config["scales"]
        
        records = []
        
        for scale in sorted(scales):
            # Apply attack perturbation to raw test features for tabular baselines
            X_adv_raw = X_raw_test.copy()
            X_adv_raw[:, feat_idx] = X_adv_raw[:, feat_idx] * scale
            X_adv_scaled = np.log1p(np.clip(X_adv_raw, a_min=0.0, a_max=None))
            
            # --- EVALUATE TABULAR BASELINES ---
            for name, model in baselines.items():
                y_pred = model.predict(X_adv_scaled)
                y_prob = None
                if hasattr(model, "predict_proba"):
                    try:
                        y_prob = model.predict_proba(X_adv_scaled)[:, 1]
                    except Exception:
                        y_prob = None
                        
                metrics = compute_all_metrics(y_true, y_pred, y_prob)
                records.append({
                    "Scale": f"{scale}x",
                    "Model": name,
                    **metrics
                })

            # --- EVALUATE GNN (MFGraph) ---
            gnn_probs = []
            with torch.no_grad():
                for batch in test_loader:
                    batch_x_adv = batch.x.clone()
                    batch_x_adv[:, feat_idx] = batch_x_adv[:, feat_idx] * scale
                    
                    out = gnn_model(batch_x_adv.to(device), batch.edge_index.to(device), batch.batch.to(device))
                    probs = torch.sigmoid(out.view(-1)).cpu().numpy()
                    gnn_probs.extend(probs)
                    
            gnn_probs = np.array(gnn_probs)
            gnn_preds = (gnn_probs >= 0.5).astype(int)
            gnn_metrics = compute_all_metrics(y_true, gnn_preds, gnn_probs)
            
            records.append({
                "Scale": f"{scale}x",
                "Model": "GNN (MFGraph)",
                **gnn_metrics
            })
            
            print(f"Scale {scale}x | GNN F1: {gnn_metrics['F1']:.4f} | Baselines Evaluated: {len(baselines)}")

        # 4. Save and Summarize (Matching PE script output formatting)
        df_all = pd.DataFrame(records)
        
        # Detailed long-format CSV for multi-metric plots (e.g. Recall and FPR)
        df_all.to_csv(out_dir / f"{attack_name}_multimetric.csv", index=False)
        
        # Pivot tables
        df_f1 = df_all.pivot(index="Scale", columns="Model", values="F1")
        df_f1.to_csv(out_dir / f"{attack_name.replace('_', ' ')}_robustness.csv")
        
        df_rec = df_all.pivot(index="Scale", columns="Model", values="Recall")
        df_rec.to_csv(out_dir / f"{attack_name.replace('_', ' ')}_recall.csv")
        
        print("\n" + "="*50)
        print(f"      F1-SCORE DEGRADATION: {attack_name.replace('_', ' ')}")
        print("="*50)
        print(df_f1.round(4))
        
        print("\n" + "="*50)
        print(f"    MALWARE RECALL DEGRADATION: {attack_name.replace('_', ' ')}")
        print("="*50)
        print(df_rec.round(4))
        print(f"\n[Success] All results exported to {out_dir}")

if __name__ == "__main__":
    evaluate_nids_adversarial()