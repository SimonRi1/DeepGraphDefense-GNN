import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
import joblib
import warnings

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.training.nids_dataset import get_nids_splits
from src.utils.config import CONFIG
warnings.filterwarnings("ignore")

def graphs_to_tabular(dataloader):
    """
    Flattens each window graph into one vector (mean of its flow features), so the
    baselines see the same information as the GNN except the graph structure.
    """
    X, y = [], []
    for batch in tqdm(dataloader, desc="Flattening Graphs for Baselines", leave=False):
        pooled = global_mean_pool(batch.x, batch.batch)
        X.append(pooled.numpy())
        y.extend(batch.y.numpy())
    return np.vstack(X), np.array(y)

def train_and_save_baselines():
    data_dir = project_root / CONFIG["cic_graphs_path"]

    # Same chronological split as the NIDS GNN, so both are tested on the same unseen future windows
    train_loader, _, _ = get_nids_splits(data_dir, batch_size=256, seed=42)
    train_loader = DataLoader(train_loader.dataset, batch_size=256, shuffle=False)

    print(f"\n[NIDS Baselines] Processing {len(train_loader.dataset)} training graphs for NIDS baselines...")
    X_train_raw, y_train = graphs_to_tabular(train_loader)

    # Node features are already log1p-scaled by cic_graph.py: use the pooled values as they are,
    # exactly like the GNN (the evaluation scripts do the same)
    X_train = X_train_raw

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree (ID3)": DecisionTreeClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=50, random_state=42),
        "SVM": SVC(probability=True, random_state=42)
    }

    baseline_dir = project_root / "experiments" / "nids_baselines_training"
    baseline_dir.mkdir(exist_ok=True, parents=True)

    print("\n[NIDS Baselines] Training and saving models...")
    for name, model in models.items():
        print(f" -> Training {name}...")
        model.fit(X_train, y_train)
        
        # Saved as joblib for the drift / adversarial evaluation scripts (name must match there)
        model_filename = baseline_dir / f"{name.replace(' ', '_').replace('(', '').replace(')', '')}.joblib"
        joblib.dump(model, model_filename)

    print(f"\n[Success] All NIDS baseline models trained and saved to {baseline_dir}.")

if __name__ == "__main__":
    train_and_save_baselines()