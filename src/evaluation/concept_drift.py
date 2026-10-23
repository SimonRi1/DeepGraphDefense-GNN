"""
Concept-drift evaluation: models trained on the past, scored on later periods.
  - PE:   trained on January 2018, scored on each month Feb-Dec 2018 (EMBER test/<MM>/)
  - NIDS: scored on 5 chronological chunks of the held-out last 10% of every capture
Output: experiments/concept_drift/{PE,NIDS}_concept_drift_*.csv / .png
"""
import sys
import os
from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import roc_auc_score, f1_score, recall_score
from sklearn.svm import SVC
import joblib
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.config import CONFIG
from src.models.gnn import MFGraph
from src.training.pe_dataset import PEGraphDataset

# Parallel workers for reading .pt graph files (I/O bound)
NUM_WORKERS = min(8, os.cpu_count() or 1)


# =====================================================================
# Model Checkpoint Resolvers
# =====================================================================

def load_latest_gnn(domain="pe", device="cpu"):
    """Loads the GNN model from the latest timestamped folder for PE or NIDS."""
    gnn_config = CONFIG.get("gnn", {"hidden_dim": 64, "k": 10, "dropout_rate": 0.5})
    base_dir = project_root / CONFIG.get("experiments_path", "experiments")
    exp_dir = base_dir / ("pe_gnn_training" if domain == "pe" else "nids_gnn_training")

    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment folder not found: {exp_dir}")

    timestamp_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir()], reverse=True)
    if not timestamp_dirs:
        raise FileNotFoundError(f"No timestamped folders found in {exp_dir}")

    latest_run = timestamp_dirs[0]
    checkpoints = list(latest_run.rglob("*.pth")) + list(latest_run.rglob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {latest_run}")

    weights_path = max(checkpoints, key=lambda f: f.stat().st_mtime)
    print(f"[{domain.upper()} Drift] Loaded GNN: {latest_run.name}/{weights_path.name}")

    input_dim = CONFIG["max_feature_len"] if domain == "pe" else CONFIG["nids"]["input_dim"]
    model = MFGraph(
        input_dim=input_dim,
        hidden_dim=gnn_config.get("hidden_dim", 64),
        k=gnn_config.get("k", 10),
        dropout_rate=gnn_config.get("dropout_rate", 0.5)
    ).to(device)

    state_dict = torch.load(weights_path, map_location=device)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _baseline_predict(model, X):
    """Returns (labels, scores) for a tabular baseline.
    SVC: one decision_function pass (sign == predict, same ranking as predict_proba for AUC),
    instead of two slow kernel passes."""
    if isinstance(model, SVC):
        scores = model.decision_function(X)
        return (scores > 0).astype(int), scores
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
    return y_pred, y_prob


def load_joblib_baselines(baseline_dir, model_names):
    """Loads all pre-trained tabular baselines from .joblib files."""
    baselines = {}
    if not baseline_dir.exists():
        print(f"[Warning] Baseline directory not found: {baseline_dir}")
        return baselines

    for name in model_names:
        clean_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        path = baseline_dir / f"{clean_name}.joblib"
        if path.exists():
            baselines[name] = joblib.load(path)
        else:
            print(f"[Warning] Missing baseline: {path.name}")
    return baselines


# =====================================================================
# PE Domain Concept Drift (Monthly EMBER Subfolders: Months 02-12)
# =====================================================================

def evaluate_pe_concept_drift(device):
    print("\n========================================================")
    print("      PE DOMAIN: CONCEPT DRIFT CHRONOLOGICAL TEST")
    print("========================================================")

    test_base_path = project_root / "data" / "graphs" / "ember_graphs" / "test"
    monthly_loaders = {}

    for month in range(2, 13):
        m_path = test_base_path / f"{month:02d}"
        if m_path.exists() and list(m_path.rglob("*.pt")):
            ds = PEGraphDataset(str(m_path))
            # Evaluation batching
            monthly_loaders[month] = DataLoader(ds, batch_size=512, shuffle=False, num_workers=NUM_WORKERS)

    if not monthly_loaders:
        raise FileNotFoundError(f"No monthly subdirectories found in {test_base_path}")

    # Load Models
    gnn_model = load_latest_gnn(domain="pe", device=device)
    baseline_dir = project_root / "experiments" / "pe_baselines_training"
    pe_models = [
        "LR-based", "DT-based", "SVM-based", "KNN-based",
        "AdaBoost-based", "MLP-based", "FFDNN-based"
    ]
    baselines = load_joblib_baselines(baseline_dir, pe_models)

    records = []

    for month, loader in monthly_loaders.items():
        print(f"\nEvaluating Month {month:02d} ({len(loader.dataset)} graphs)...")
        all_pooled = []
        labels = []
        gnn_probs = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Month {month:02d}", leave=False):
                # 1. GNN Prediction
                out = gnn_model(batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device) if hasattr(batch, 'batch') else None)
                probs = torch.sigmoid(out.view(-1)).cpu().numpy()
                gnn_probs.extend(probs)

                # 2. Extract Pooled Tabular Features for Baselines
                pooled = global_mean_pool(batch.x, batch.batch if hasattr(batch, 'batch') else torch.zeros(batch.x.size(0), dtype=torch.long))
                all_pooled.append(pooled.cpu().numpy())

                # Collect labels
                y = batch.y.cpu().numpy() if hasattr(batch, 'y') else [0] * pooled.size(0)
                labels.extend(y if len(y) == pooled.size(0) else [y[0]] * pooled.size(0))

        X_tabular = np.vstack(all_pooled)
        y_true = np.array(labels, dtype=int)

        # Record GNN Performance
        gnn_preds = (np.array(gnn_probs) >= 0.5).astype(int)
        gnn_auc = roc_auc_score(y_true, gnn_probs) if len(np.unique(y_true)) > 1 else np.nan
        records.append({
            "TimeStep": f"Month_{month:02d}",
            "Month": month,
            "Model": "GNN (MFGraph)",
            "AUC": gnn_auc,
            "F1": f1_score(y_true, gnn_preds, zero_division=0),
            "Recall": recall_score(y_true, gnn_preds, zero_division=0)
        })

        # Record Tabular Baselines Performance
        for name, model in baselines.items():
            y_pred, y_prob = _baseline_predict(model, X_tabular)
            auc = roc_auc_score(y_true, y_prob) if y_prob is not None and len(np.unique(y_true)) > 1 else np.nan

            records.append({
                "TimeStep": f"Month_{month:02d}",
                "Month": month,
                "Model": name,
                "AUC": auc,
                "F1": f1_score(y_true, y_pred, zero_division=0),
                "Recall": recall_score(y_true, y_pred, zero_division=0)
            })

    # Save & Plot PE Results
    out_dir = project_root / CONFIG.get("experiments_path", "experiments") / "concept_drift"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_pe = pd.DataFrame(records)
    df_pe.to_csv(out_dir / "PE_concept_drift_metrics.csv", index=False)

    df_pivot_auc = df_pe.pivot(index="Month", columns="Model", values="AUC")
    df_pivot_auc.to_csv(out_dir / "PE_concept_drift_AUC.csv")

    _generate_drift_plot(
        df_pe,
        x_col="Month",
        metric="AUC",
        title="Host Domain: Concept Drift on Malware Detection (EMBER 2018)",
        xlabel="Evaluation Month (2018)",
        save_path=out_dir / "PE_concept_drift_AUC.png"
    )
    print(f"[Success] PE Drift evaluation complete. Artifacts saved to: {out_dir}")


# =====================================================================
# NIDS Domain Concept Drift (Sequential Temporal Chunks: Chunks 1-10)
# =====================================================================

def evaluate_nids_concept_drift(device):
    print("\n========================================================")
    print("     NIDS DOMAIN: CONCEPT DRIFT CHRONOLOGICAL TEST")
    print("========================================================")

    nids_data_path = project_root / "data" / "graphs" / "nids_graphs"
    if not nids_data_path.exists():
        raise FileNotFoundError(f"NIDS graph directory not found: {nids_data_path}")

    dataset = PEGraphDataset(str(nids_data_path))
    num_chunks = 5
    chunk_indices = _nids_heldout_chunks(nids_data_path, num_chunks)
    print(f"[Info] Evaluating held-out test windows across {num_chunks} chronological chunks "
          f"({[len(c) for c in chunk_indices]} graphs)...")

    # Load Models
    gnn_model = load_latest_gnn(domain="nids", device=device)
    baseline_dir = project_root / "experiments" / "nids_baselines_training"
    nids_models = ["Random Forest", "Decision Tree ID3", "AdaBoost", "MLP", "SVM"]
    baselines = load_joblib_baselines(baseline_dir, nids_models)

    records = []

    for chunk_idx, indices in enumerate(chunk_indices):
        chunk_subset = torch.utils.data.Subset(dataset, indices)
        loader = DataLoader(chunk_subset, batch_size=512, shuffle=False, num_workers=NUM_WORKERS)

        all_pooled = []
        labels = []
        gnn_probs = []

        with torch.no_grad():
            for batch in loader:
                # 1. GNN Inference
                out = gnn_model(batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device))
                probs = torch.sigmoid(out.view(-1)).cpu().numpy()
                gnn_probs.extend(probs)

                # 2. Extract Tabular Features
                pooled = global_mean_pool(batch.x, batch.batch)
                all_pooled.append(pooled.cpu().numpy())

                y = batch.y.cpu().numpy() if hasattr(batch, 'y') else [0] * pooled.size(0)
                labels.extend(y if len(y) == pooled.size(0) else [y[0]] * pooled.size(0))

        # Pooled features are already log1p-scaled (same input as baseline training)
        X_scaled = np.vstack(all_pooled)
        y_true = np.array(labels, dtype=int)

        chunk_label = f"Chunk_{chunk_idx + 1:02d}"

        # Evaluate GNN
        gnn_preds = (np.array(gnn_probs) >= 0.5).astype(int)
        gnn_auc = roc_auc_score(y_true, gnn_probs) if len(np.unique(y_true)) > 1 else np.nan
        records.append({
            "TimeStep": chunk_label,
            "Chunk": chunk_idx + 1,
            "Model": "GNN (MFGraph)",
            "AUC": gnn_auc,
            "F1": f1_score(y_true, gnn_preds, zero_division=0),
            "Recall": recall_score(y_true, gnn_preds, zero_division=0)
        })

        # Evaluate Tabular Baselines
        for name, model in baselines.items():
            y_pred, y_prob = _baseline_predict(model, X_scaled)
            auc = roc_auc_score(y_true, y_prob) if y_prob is not None and len(np.unique(y_true)) > 1 else np.nan

            records.append({
                "TimeStep": chunk_label,
                "Chunk": chunk_idx + 1,
                "Model": name,
                "AUC": auc,
                "F1": f1_score(y_true, y_pred, zero_division=0),
                "Recall": recall_score(y_true, y_pred, zero_division=0)
            })

    # Save & Plot NIDS Results
    out_dir = project_root / CONFIG.get("experiments_path", "experiments") / "concept_drift"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_nids = pd.DataFrame(records)
    df_nids.to_csv(out_dir / "NIDS_concept_drift_metrics.csv", index=False)

    df_pivot_f1 = df_nids.pivot(index="Chunk", columns="Model", values="F1")
    df_pivot_f1.to_csv(out_dir / "NIDS_concept_drift_F1.csv")

    _generate_drift_plot(
        df_nids,
        x_col="Chunk",
        metric="F1",
        title="Network Domain: Concept Drift Over Held-out Temporal Chunks",
        xlabel="Time Chunk (Chronological Slices of the Held-out 10% per Scenario)",
        save_path=out_dir / "NIDS_concept_drift_F1.png"
    )
    print(f"[Success] NIDS Drift evaluation complete. Artifacts saved to: {out_dir}")


def _nids_heldout_chunks(data_dir, num_chunks):
    """
    Returns num_chunks lists of dataset indices covering only the held-out test
    windows (last 10% of each scenario, same boundaries as get_nids_splits).
    Chunk k gathers the k-th chronological slice of every scenario's test part.
    """
    # Sorted like PEGraphDataset.file_paths, so position == dataset index
    files = sorted(p.name for p in Path(data_dir).rglob('*.pt'))

    scenario_groups = {}
    for idx, f in enumerate(files):
        scenario_groups.setdefault(f.split('_window_')[0], []).append(idx)

    chunks = [[] for _ in range(num_chunks)]
    for indices in scenario_groups.values():
        test_part = indices[int(0.90 * len(indices)):]
        for k, part in enumerate(np.array_split(test_part, num_chunks)):
            chunks[k].extend(int(i) for i in part)
    return chunks


# =====================================================================
# Shared Plotting Helper
# =====================================================================

def _generate_drift_plot(df, x_col, metric, title, xlabel, save_path):
    plt.figure(figsize=(12, 6))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
    models = df['Model'].unique()

    for idx, model in enumerate(models):
        m_df = df[df['Model'] == model].sort_values(by=x_col)
        is_gnn = "GNN" in model or "MFGraph" in model
        linewidth = 3.5 if is_gnn else 1.5

        # Calculate Degradation Rate: (Max - Min) / Max
        val_max = m_df[metric].max()
        val_min = m_df[metric].min()
        deg = ((val_max - val_min) / val_max * 100) if (pd.notnull(val_max) and val_max > 0) else 0.0

        label = f"{model} (Deg: {deg:.1f}%)" if is_gnn else model

        plt.plot(
            m_df[x_col],
            m_df[metric],
            marker=markers[idx % len(markers)],
            linewidth=linewidth,
            label=label
        )

    plt.title(title, fontsize=13, fontweight='bold', pad=15)
    plt.xlabel(xlabel, fontsize=11)
    plt.ylabel(f"{metric} Score", fontsize=11)
    plt.ylim(0.4, 1.05)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# =====================================================================
# Entrypoint
# =====================================================================

def run_concept_drift():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Concept Drift] Initializing on device: {device}")
    
    # 1. Run PE Domain Chronological Drift
    evaluate_pe_concept_drift(device)
    
    # 2. Run NIDS Domain Temporal Drift
    evaluate_nids_concept_drift(device)


if __name__ == "__main__":
    run_concept_drift()