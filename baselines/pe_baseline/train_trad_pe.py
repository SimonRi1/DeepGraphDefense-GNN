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
from lightgbm import LGBMClassifier

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.training.pe_dataset import PEGraphDataset

def extract_flat_features(loader):
    X, y = [], []
    for data in loader:
        X.append(data.x.mean(dim=0).numpy())
        y.append(data.y.item())
    return np.array(X), np.array(y)

def train_pe_baselines():
    print("[PE Baselines] Loading training data...")
    
    pe_train_path = project_root / "data" / "graphs"  / "train" 
    pe_dataset = PEGraphDataset(str(pe_train_path))
    
    # avoiding RAM saturation
    subset_size = min(10000, len(pe_dataset)) 
    pe_subset = torch.utils.data.Subset(pe_dataset, range(subset_size))
    pe_loader = DataLoader(pe_subset, batch_size=1, shuffle=True)
    
    X_train, y_train = extract_flat_features(pe_loader)
    
    baselines = {
        "LR-based": LogisticRegression(max_iter=1000),
        "DT-based": DecisionTreeClassifier(),
        "SVM-based": SVC(probability=True),
        "KNN-based": KNeighborsClassifier(),
        "AdaBoost-based": AdaBoostClassifier(),
        "MLP-based": MLPClassifier(max_iter=500),
        "LightGBM": LGBMClassifier(n_estimators=100)
    }
    
    out_dir = project_root / "experiments" / "PEbaselines_training"
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