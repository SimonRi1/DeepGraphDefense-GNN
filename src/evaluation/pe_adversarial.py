"""
PE evasion test: malware test graphs are moved towards GAN-generated malware variants
(x' = x + scale * (G(z) - x)), benign graphs are untouched, so recall is the key metric.
Scale 0.0 is the clean reference. Output: experiments/pe_adversarial/*.csv
"""
import sys
import os
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader
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

# Traditional Baselines
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

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
    tn, fp, fn, tp = cm.ravel()
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


def load_pe_gnn(device):
    """Loads the trained PE MFGraph."""
    gnn_config = CONFIG["gnn"]
    pe_dir = project_root / CONFIG["experiments_path"] / "pe_gnn_training"
    
    # Find all subdirectories and sort them descending (latest timestamp first)
    timestamp_folders = sorted([d for d in pe_dir.iterdir() if d.is_dir()], reverse=True)
    
    if not timestamp_folders:
        raise FileNotFoundError(f"No timestamped folders found in {pe_dir}")
        
    # Select the most recent folder
    latest_run = timestamp_folders[0]
    
    # Search for checkpoints only inside the latest timestamped folder
    checkpoints = list(latest_run.rglob("*.pth")) + list(latest_run.rglob("*.pt"))
    
    if not checkpoints:
        raise FileNotFoundError(f"No PE GNN checkpoint found in {latest_run}")
        
    pe_weights = max(checkpoints, key=lambda f: f.stat().st_mtime)
    print(f"[PE Adversarial] Loaded GNN weights from: {latest_run.name}/{pe_weights.name}")
    
    model = MFGraph(
        input_dim=512,
        hidden_dim=gnn_config.get("hidden_dim", 64),
        k=gnn_config.get("k", 10),
        dropout_rate=gnn_config.get("dropout_rate", 0.5)
    ).to(device)
    
    state_dict = torch.load(pe_weights, map_location=device)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_pe_gan(device):
    """
    Loads the trained PE GraphGenerator (generator.pth) from the latest GAN run.
    Architecture sizes and the feature range are recovered from the state dict.
    """
    gan_dir = project_root / CONFIG["experiments_path"] / "pe_gan_training"
    checkpoints = list(gan_dir.rglob("generator.pth")) if gan_dir.exists() else []
    if not checkpoints:
        print(f"[PE Adversarial] WARNING: no generator.pth in {gan_dir}. Falling back to Gaussian noise.")
        return None

    gen_path = max(checkpoints, key=lambda f: f.stat().st_mtime)
    state = torch.load(gen_path, map_location=device)

    from src.models.gan import GraphGenerator
    num_nodes, hidden_dim = state["pos_emb.weight"].shape
    feat_min = state["feat_min"]
    generator = GraphGenerator(
        noise_dim=state["inp.weight"].shape[1] // 2,
        output_dim=state["out.weight"].shape[0],
        num_nodes=num_nodes,
        feat_min=feat_min,
        feat_max=feat_min + state["feat_range"],
        hidden_dim=hidden_dim,
    ).to(device)
    generator.load_state_dict(state)
    generator.eval()
    print(f"[PE Adversarial] Loaded GAN generator from: {gen_path.parent.name}/{gen_path.name}")
    return generator


def extract_flat_features(loader):
    """Averages node features to form a 512-dim tabular vector per PE."""
    X, y = [], []
    for data in loader:
        X.append(data.x.mean(dim=0).cpu().numpy())
        label = data.y.item() if hasattr(data.y, "item") else int(data.y)
        y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def load_graphs(loader):
    """Materialises the evaluation subset once so every scale perturbs the same graphs."""
    graphs = [data for data in loader]
    labels = np.array([int(g.y.item()) for g in graphs], dtype=np.int32)
    return graphs, labels


@torch.no_grad()
def perturb_graph(data, generator, scale, device):
    """
    Returns a copy of a malicious graph with perturbed node features.
      - GAN mode:   interpolates towards a GAN-generated malware variant on the same
                    topology, x' = x + scale * (G(z) - x), kept inside the real feature range.
      - Noise mode: Gaussian noise scaled by the per-feature std across nodes (fallback).
    """
    adv = data.clone()
    x = data.x.to(device)
    if generator is not None:
        batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
        fake = generator(data.edge_index.to(device), batch)
        adv_x = x + scale * (fake - x)
        lo, hi = generator.feat_min, generator.feat_min + generator.feat_range
        if lo.size(0) > 1:
            lo, hi = lo[:x.size(0)], hi[:x.size(0)]
        adv_x = torch.max(torch.min(adv_x, hi), lo)
    else:
        node_std = x.std(dim=0, keepdim=True)
        node_std[node_std < 1e-4] = 1.0
        adv_x = (x + torch.randn_like(x) * node_std * (scale * 2.0)).clamp(min=0.0)
    adv.x = adv_x.cpu()
    return adv


def evaluate_pe_adversarial():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n========================================================")
    print(f"   PE ADVERSARIAL STRESS TEST (HOST DOMAIN)")
    print(f"========================================================")
    print(f"[Info] Running on device: {device}")
    
    # 1. Load Data across all test monthly subfolders
    pe_test_path = project_root / "data" / "graphs" / "ember_graphs" / "test"
    monthly_datasets = []
    
    for month in range(2, 13):
        m_path = pe_test_path / f"{month:02d}"
        if m_path.exists() and next(m_path.rglob("*.pt"), None):
            monthly_datasets.append(PEGraphDataset(str(m_path)))
            
    if not monthly_datasets:
        # Fallback to root test folder if flat
        if pe_test_path.exists() and next(pe_test_path.rglob("*.pt"), None):
            monthly_datasets.append(PEGraphDataset(str(pe_test_path)))
        else:
            raise FileNotFoundError(f"No test graphs found in {pe_test_path}")
            
    pe_dataset = ConcatDataset(monthly_datasets)
    print(f"[Info] Total test graphs found: {len(pe_dataset)}")
    
    subset_size = min(2000, len(pe_dataset))
    random.seed(42)
    indices = list(range(len(pe_dataset)))
    random.shuffle(indices)
    pe_subset = torch.utils.data.Subset(pe_dataset, indices[:subset_size])
    pe_loader = DataLoader(pe_subset, batch_size=1, shuffle=False)
    
    graphs, y_true = load_graphs(pe_loader)
    malware_mask = (y_true > 0.5)
    print(f"[Info] Sample split -> Malware: {np.sum(malware_mask)} | Benign: {len(y_true) - np.sum(malware_mask)}")
    
    # 2. Load Models
    gnn_model = load_pe_gnn(device)
    gan_gen = load_pe_gan(device)
    
    baseline_dir = project_root / "experiments" / "pe_baselines_training"
    model_names = ["LR-based", "DT-based", "SVM-based", "KNN-based", "AdaBoost-based", "MLP-based", "FFDNN-based"]
    baselines = {}
    
    if baseline_dir.exists():
        for name in model_names:
            model_path = baseline_dir / f"{name.replace(' ', '_')}.joblib"
            if model_path.exists():
                baselines[name] = joblib.load(model_path)

    if not baselines:
        raise FileNotFoundError(f"No pre-trained PE baseline models found in {baseline_dir}. Run baseline training first.")


    # 3. Adversarial Stress Test Loop
    perturbation_scales = [0.0, 0.25, 0.5, 0.75, 1.0]
    records = []

    attack_mode = "GAN" if gan_gen is not None else "Gaussian"
    X_clean, _ = extract_flat_features(pe_loader)
    torch.manual_seed(42)

    for scale in perturbation_scales:
        print(f"\n[Testing Intensity: {scale}x | Mode: {attack_mode}]")
        
        # Perturb malware graphs only (simulating evasion attempt); every model sees the same graphs
        if scale > 0.0:
            adv_graphs = [perturb_graph(g, gan_gen, scale, device) if mal else g
                          for g, mal in zip(graphs, malware_mask)]
        else:
            adv_graphs = graphs
        adv_loader = DataLoader(adv_graphs, batch_size=1, shuffle=False)
        X_test_adv, _ = extract_flat_features(adv_loader)

        if scale > 0.0:
            delta = np.mean(np.abs(X_test_adv[malware_mask] - X_clean[malware_mask]))
            print(f" -> Active Feature Shift Delta: {delta:.4f}")
        else:
            print(f" -> Baseline (Clean Data)")

        # Evaluate Tabular Baselines
        for name, model in baselines.items():
            y_pred = model.predict(X_test_adv)
            
            y_prob = None
            if hasattr(model, "predict_proba"):
                try:
                    y_prob = model.predict_proba(X_test_adv)[:, 1]
                except Exception:
                    y_prob = None
                    
            metrics = compute_all_metrics(y_true, y_pred, y_prob)
            records.append({
                "Intensity": f"{scale}x",
                "Scale": scale,
                "Mode": attack_mode,
                "Model": name,
                **metrics
            })

        # Evaluate GNN (MFGraph) on the same perturbed graphs
        gnn_probs = []
        with torch.no_grad():
            for batch in DataLoader(adv_graphs, batch_size=64, shuffle=False):
                batch = batch.to(device)
                out = gnn_model(batch.x, batch.edge_index, batch.batch)
                gnn_probs.append(torch.sigmoid(out.view(-1)).cpu().numpy())

        gnn_probs = np.concatenate(gnn_probs)
        gnn_preds = (gnn_probs >= 0.5).astype(int)
        gnn_metrics = compute_all_metrics(y_true, gnn_preds, gnn_probs)
        
        records.append({
            "Intensity": f"{scale}x",
            "Scale": scale,
            "Mode": attack_mode,
            "Model": "GNN (MFGraph)",
            **gnn_metrics
        })

    # 4. Save and Summarize
    df_all = pd.DataFrame(records)
    out_dir = project_root / "experiments" / "pe_adversarial"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Detailed long-format CSV for multi-metric plots
    df_all.to_csv(out_dir / "GAN_Evasion_multimetric.csv", index=False)
    
    # Pivot tables
    df_f1 = df_all.pivot(index="Intensity", columns="Model", values="F1")
    df_f1.to_csv(out_dir / "GAN_Evasion_baselines.csv")
    
    df_rec = df_all.pivot(index="Intensity", columns="Model", values="Recall")
    
    print("\n" + "="*50)
    print("      F1-SCORE UNDER ADVERSARIAL STRESS")
    print("="*50)
    print(df_f1.round(4))
    
    print("\n" + "="*50)
    print("    MALWARE RECALL (DETECTION RATE) DEGRADATION")
    print("="*50)
    print(df_rec.round(4))
    print(f"\n[Success] All results exported to {out_dir}")


if __name__ == "__main__":
    evaluate_pe_adversarial()