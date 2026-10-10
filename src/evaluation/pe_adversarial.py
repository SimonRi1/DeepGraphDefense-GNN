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
from lightgbm import LGBMClassifier

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
    pe_dir = project_root / CONFIG["experiments_path"] / "gnn_training"
    
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
    """Loads the trained GAN Generator if available."""
    gan_dir = project_root / CONFIG["experiments_path"] / "gan_training"
    if not gan_dir.exists():
        print(f"[PE Adversarial] GAN folder not found at {gan_dir}. Using variance-scaled perturbation.")
        return None
        
    checkpoints = list(gan_dir.rglob("*.pth")) + list(gan_dir.rglob("*.pt"))
    if not checkpoints:
        print("[PE Adversarial] No GAN checkpoint found. Using variance-scaled perturbation.")
        return None
        
    gen_ckpts = [c for c in checkpoints if "gen" in c.name.lower()]
    chosen_ckpt = max(gen_ckpts, key=lambda f: f.stat().st_mtime) if gen_ckpts else max(checkpoints, key=lambda f: f.stat().st_mtime)
    
    try:
        from src.models.gan import Generator
        generator = Generator(input_dim=512, noise_dim=128).to(device)
        loaded = torch.load(chosen_ckpt, map_location=device)
        
        if isinstance(loaded, dict):
            if "generator_state_dict" in loaded:
                generator.load_state_dict(loaded["generator_state_dict"])
            elif "generator" in loaded:
                generator.load_state_dict(loaded["generator"])
            else:
                generator.load_state_dict(loaded)
        else:
            generator.load_state_dict(loaded)
            
        generator.eval()
        print(f"[PE Adversarial] Successfully loaded GAN Generator from: {chosen_ckpt.name}")
        return generator
    except Exception as e:
        print(f"[PE Adversarial] Could not load GAN ({e}). Falling back to variance-scaled perturbation.")
        return None


def extract_flat_features(loader):
    """Averages node features to form a 512-dim tabular vector per PE."""
    X, y = [], []
    for data in loader:
        X.append(data.x.mean(dim=0).cpu().numpy())
        label = data.y.item() if hasattr(data.y, "item") else int(data.y)
        y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def apply_adversarial_perturbation(X_malware, generator, scale, device):
    """
    Applies adversarial evasion noise.
    Scales perturbation against feature variance so values reliably cross decision thresholds.
    """
    if scale == 0.0 or len(X_malware) == 0:
        return X_malware.copy()
        
    n_samples, n_features = X_malware.shape
    
    feature_std = np.std(X_malware, axis=0, keepdims=True)
    feature_std[feature_std < 1e-4] = 1.0  # Avoid zero variance
    
    noise = None
    if generator is not None:
        try:
            with torch.no_grad():
                z = torch.randn(n_samples, 128, device=device)
                gen_out = generator(z)
                noise = gen_out.cpu().numpy()
        except Exception:
            noise = None
            
    if noise is None:
        rng = np.random.RandomState(42 + int(scale * 100))
        noise = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))
        
    # Scale perturbation: 0.25x -> 0.5 std dev, 1.0x -> 2.0 std dev shift
    perturbation = noise * feature_std * (scale * 2.0)
    return X_malware + perturbation


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
    
    X_flat, y_true = extract_flat_features(pe_loader)
    malware_mask = (y_true > 0.5)
    print(f"[Info] Sample split -> Malware: {np.sum(malware_mask)} | Benign: {len(y_true) - np.sum(malware_mask)}")
    
    # 2. Load Models
    gnn_model = load_pe_gnn(device)
    gan_gen = load_pe_gan(device)
    
    baseline_dir = project_root / "experiments" / "PEbaselines_training"
    model_names = ["LR-based", "DT-based", "SVM-based", "KNN-based", "AdaBoost-based", "MLP-based", "LightGBM"]
    baselines = {}
    
    if baseline_dir.exists():
        for name in model_names:
            model_path = baseline_dir / f"{name.replace(' ', '_')}.joblib"
            if model_path.exists():
                baselines[name] = joblib.load(model_path)
                
    if not baselines:
        print("[Notice] Pre-trained baseline weights not found. Auto-fitting baselines on clean samples...")
        init_models = {
            "LR-based": LogisticRegression(max_iter=1000),
            "DT-based": DecisionTreeClassifier(random_state=42),
            "SVM-based": SVC(probability=True, random_state=42),
            "KNN-based": KNeighborsClassifier(),
            "AdaBoost-based": AdaBoostClassifier(random_state=42),
            "MLP-based": MLPClassifier(max_iter=500, random_state=42),
            "LightGBM": LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        }
        for name, mdl in init_models.items():
            mdl.fit(X_flat, y_true)
            baselines[name] = mdl

    # 3. Adversarial Stress Test Loop
    perturbation_scales = [0.0, 0.25, 0.5, 0.75, 1.0]
    records = []

    for scale in perturbation_scales:
        print(f"\n[Testing Intensity: {scale}x]")
        
        # Perturb malware features only (simulating evasion attempt)
        X_test_adv = X_flat.copy()
        if scale > 0.0 and np.any(malware_mask):
            X_test_adv[malware_mask] = apply_adversarial_perturbation(
                X_flat[malware_mask], gan_gen, scale, device
            )
            delta = np.mean(np.abs(X_test_adv[malware_mask] - X_flat[malware_mask]))
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
                "Model": name,
                **metrics
            })

        # Evaluate GNN (MFGraph)
        gnn_probs = []
        with torch.no_grad():
            for i, data in enumerate(pe_loader):
                is_mal = (y_true[i] > 0.5)
                
                if is_mal and scale > 0.0:
                    node_std = data.x.std(dim=0, keepdim=True)
                    node_std[node_std < 1e-4] = 1.0
                    
                    if gan_gen is not None:
                        try:
                            z = torch.randn(data.x.size(0), 128, device=device)
                            adv_noise = gan_gen(z)
                        except Exception:
                            adv_noise = torch.randn_like(data.x)
                    else:
                        adv_noise = torch.randn_like(data.x)
                        
                    adv_x = data.x + adv_noise.to(data.x.device) * node_std * (scale * 2.0)
                else:
                    adv_x = data.x
                    
                out = gnn_model(adv_x.to(device), data.edge_index.to(device), None)
                prob = torch.sigmoid(out.view(-1)).mean().item()
                gnn_probs.append(prob)

        gnn_probs = np.array(gnn_probs)
        gnn_preds = (gnn_probs >= 0.5).astype(int)
        gnn_metrics = compute_all_metrics(y_true, gnn_preds, gnn_probs)
        
        records.append({
            "Intensity": f"{scale}x",
            "Scale": scale,
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