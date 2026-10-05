import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from src.training.nids_dataset import get_nids_splits
import warnings

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.training.pe_dataset import PEGraphDataset
warnings.filterwarnings("ignore")

def graphs_to_tabular(dataloader):
    """Collapses RAW NIDS time-window graphs into 1D vectors using mean pooling."""
    X, y = [], []
    for batch in tqdm(dataloader, desc="Flattening Graphs for Baselines", leave=False):
        # Mean pool the RAW flow features across the time window
        pooled = global_mean_pool(batch.x, batch.batch)
        X.append(pooled.numpy())
        y.extend(batch.y.numpy())
    return np.vstack(X), np.array(y)

def test_adversarial_tabular(model, X_test_raw, y_test, feature_idx, scales, model_name):
    """Applies multiplier to RAW features, then scales, then predicts."""
    results = {}
    baseline_f1 = None
    
    for scale in scales:
        # 1. Clone RAW features and apply the attack multiplier
        X_adv_raw = X_test_raw.copy()
        X_adv_raw[:, feature_idx] = X_adv_raw[:, feature_idx] * scale
        
        # 2. Apply the exact same scaling step used by the GNN
        X_adv_scaled = np.log1p(np.clip(X_adv_raw, a_min=0.0, a_max=None))
        
        # 3. Predict
        preds = model.predict(X_adv_scaled)
        f1 = f1_score(y_test, preds, zero_division=0)
        results[f"{scale}x"] = f1
        
        if scale == 1.0:
            baseline_f1 = f1
            
    worst_f1 = min(results.values())
    if baseline_f1 and baseline_f1 > 0:
        degradation = (baseline_f1 - worst_f1) / baseline_f1
    else:
        degradation = 0.0
        
    results['Degradation'] = f"{degradation:.2%}"
    return results

def train_and_evaluate_baselines():
    data_dir = project_root / "data" / "graphs" / "nids_graphs"
    dataset = PEGraphDataset(str(data_dir))
    
    # 80/20 Split (Fixed seed for reproducible baseline comparison)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_data, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

    print(f"\n[Baselines] Processing {len(dataset)} graphs...")
    X_train_raw, y_train = graphs_to_tabular(train_loader)
    X_test_raw, y_test = graphs_to_tabular(test_loader)

    # Apply scaling for training
    X_train_scaled = np.log1p(np.clip(X_train_raw, a_min=0.0, a_max=None))

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree (ID3)": DecisionTreeClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=50, random_state=42)
    }

    attacks = {
        "Packet Size Evasion": {"feat_idx": 2, "scales": [1.0, 0.8, 0.5, 0.1, 2.0, 5.0]},
        "Inter-Arrival Time Evasion": {"feat_idx": 4, "scales": [1.0, 2.0, 5.0, 10.0]}
    }
    
    final_reports = {}

    for name, model in models.items():
        print(f"\n[Training] {name}...")
        model.fit(X_train_scaled, y_train)
        
        for attack_name, config in attacks.items():
            if attack_name not in final_reports:
                final_reports[attack_name] = pd.DataFrame(
                    index=models.keys(), 
                    columns=[f"{s}x" for s in config["scales"]] + ["Degradation"]
                )
                
            res = test_adversarial_tabular(model, X_test_raw, y_test, config["feat_idx"], config["scales"], name)
            for k, v in res.items():
                if k != 'Degradation':
                    final_reports[attack_name].loc[name, k] = f"{v:.4f}"
                else:
                    final_reports[attack_name].loc[name, k] = v

    # Output and Save Tables
    for attack_name, df in final_reports.items():
        print(f"\n{'='*60}")
        print(f"Adversarial Robustness Comparison: {attack_name}")
        print(f"{'='*60}")
        print(df.to_string())
        
        # Save for thesis plotting
        out_path = project_root / "experiments" / "nids_gnn" / f"{attack_name.replace(' ', '_')}_baselines.csv"
        out_path.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(out_path)
        print(f"-> Saved to {out_path}")

if __name__ == "__main__":
    train_and_evaluate_baselines()