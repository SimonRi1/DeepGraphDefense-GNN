import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
import joblib
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
        pooled = global_mean_pool(batch.x, batch.batch)
        X.append(pooled.numpy())
        y.extend(batch.y.numpy())
    return np.vstack(X), np.array(y)

def train_and_save_baselines():
    data_dir = project_root / "data" / "graphs" / "nids_graphs"
    dataset = PEGraphDataset(str(data_dir))
    
    # 80/20 Split (Fixed seed for reproducible baseline comparison)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_data, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

    print(f"\n[Baselines] Processing {len(dataset)} graphs for NIDS baselines...")
    X_train_raw, y_train = graphs_to_tabular(train_loader)

    # Apply log1p scaling for training
    X_train_scaled = np.log1p(np.clip(X_train_raw, a_min=0.0, a_max=None))

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree (ID3)": DecisionTreeClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=50, random_state=42)
    }

    baseline_dir = project_root / "experiments" / "NIDSbaselines_training"
    baseline_dir.mkdir(exist_ok=True, parents=True)

    for name, model in models.items():
        print(f"[Training] Fitting {name} on clean NIDS data...")
        model.fit(X_train_scaled, y_train)
        
        # Save model to disk so adversarial evaluation can load it
        model_filename = baseline_dir / f"{name.replace(' ', '_').replace('(', '').replace(')', '')}.joblib"
        joblib.dump(model, model_filename)
        print(f" -> Saved to {model_filename}")

    print(f"\n[Success] All NIDS baseline models trained and saved successfully.")

if __name__ == "__main__":
    train_and_save_baselines()