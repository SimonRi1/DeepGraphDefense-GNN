import sys
import numpy as np
from pathlib import Path
import joblib
import torch
from torch_geometric.loader import DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.training.pe_dataset import PEGraphDataset
from src.utils.config import CONFIG

def extract_flat_features(loader):
    """Concatenates the 9 PE nodes -> one 9*512-dim vector per file (graph structure removed).
    A mean over nodes would mix unrelated families sharing the same position."""
    X, y = [], []
    for data in loader:
        X.append(data.x.flatten().numpy())
        y.append(data.y.item())
    return np.array(X), np.array(y)

def train_pe_baselines():
    print("[PE Baselines] Loading training data...")

    graphs_dir = project_root / CONFIG["ember_graphs_path"]
    pe_train_path = graphs_dir / "train" 
    pe_dataset = PEGraphDataset(str(pe_train_path))
    
    # Seeded random 10k January graphs (not filesystem order) to keep SVM/KNN training tractable
    subset_size = min(10000, len(pe_dataset))
    indices = np.random.default_rng(CONFIG["random_seed"]).permutation(len(pe_dataset))[:subset_size]
    pe_subset = torch.utils.data.Subset(pe_dataset, indices.tolist())
    pe_loader = DataLoader(pe_subset, batch_size=1, shuffle=True)
    
    X_train, y_train = extract_flat_features(pe_loader)
    
    baselines = {
        # Scale-sensitive models get a StandardScaler (trees do not need one)
        "LR-based": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "DT-based": DecisionTreeClassifier(random_state=42),
        "SVM-based": make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
        "KNN-based": make_pipeline(StandardScaler(), KNeighborsClassifier()),
        "AdaBoost-based": AdaBoostClassifier(random_state=42),
        "MLP-based": make_pipeline(StandardScaler(), MLPClassifier(max_iter=500, random_state=42)),
        "FFDNN-based": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=42))
    }
    
    out_dir = project_root / "experiments" / "pe_baselines_training"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[PE Baselines] Training and saving models...")
    for name, model in baselines.items():
        print(f" -> Training {name}...")
        model.fit(X_train, y_train)
        
        model_path = out_dir / f"{name.replace(' ', '_')}.joblib"
        joblib.dump(model, model_path)
        
    print(f"\n[Success] All baseline models saved to {out_dir}")

if __name__ == "__main__":
    train_pe_baselines()