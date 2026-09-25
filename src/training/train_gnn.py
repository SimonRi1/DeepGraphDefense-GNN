import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
import csv
import warnings

from pathlib import Path
import sys
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.config import CONFIG
from src.utils.logger import ExperimentLogger
from src.models.gnn import MalwareGNN

gnn_config = CONFIG["gnn"]
torch.serialization.add_safe_globals([GlobalStorage, DataEdgeAttr, DataTensorAttr]) # Explicitly allowlist PyTorch Geometric internals for secure unpickling
warnings.filterwarnings("ignore")

class PEGraphDataset(Dataset):
    """Loads pre-processed PyTorch Geometric .pt files from disk."""
    def __init__(self, root_dir):
        super().__init__()
        print(f"[Dataset] Indexing files in {root_dir}...")
        
        self.file_paths = [Path(root_dir) / f for f in os.listdir(root_dir) if f.endswith('.pt')]
        
        if not self.file_paths:
            raise FileNotFoundError(f"No .pt files found in {root_dir}")
            
        print(f"[Dataset] Successfully indexed {len(self.file_paths)} graph files.")

    def len(self):
        return len(self.file_paths)

    def get(self, idx):
        return torch.load(self.file_paths[idx], weights_only=False)


def train_gnn():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    logger = ExperimentLogger(experiment_name="gnn_training", config=CONFIG)

    graphs_dir = project_root / CONFIG["graphs_path"]
    
    print("\n[GNN] Loading dataset...")
    dataset = PEGraphDataset(graphs_dir)
    
    # 80/20 Train/Test Split
    train_size = int(CONFIG["train_split"] * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # dynamic worker allocation formula
    max_cores = os.cpu_count() or 4
    optimal_workers = min(8, max_cores - 2)
    optimal_workers = max(0, optimal_workers) 
    
    print(f"[Info] Total CPU cores: {max_cores}. Allocating {optimal_workers} for data loading.\n")
    use_persistent = True if optimal_workers > 0 else False
    
    train_loader = DataLoader(train_dataset, batch_size=gnn_config["batch_size"], shuffle=True, num_workers=optimal_workers, pin_memory=True, persistent_workers=use_persistent)
    test_loader = DataLoader(test_dataset, batch_size=gnn_config["batch_size"], shuffle=False, num_workers=optimal_workers, pin_memory=True, persistent_workers=use_persistent)
    
    # Dynamically detect input_dim from the first graph
    sample_graph = dataset[0]
    input_dim = sample_graph.x.shape[1]
    print(f"[GNN] Detected input dimension: {input_dim}\n")
    
    model = MalwareGNN(
        input_dim=input_dim, 
        hidden_dim=gnn_config["hidden_dim"]
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    epochs = CONFIG["mlp"]["num_epochs"]
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            data = data.to(device)
            
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            
            loss = criterion(out.squeeze(), data.y.float())
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * data.num_graphs
            
        train_loss /= len(train_dataset)
        
        # --- Evaluation Phase ---
        model.eval()
        test_loss = 0.0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for data in tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Test]"):
                data = data.to(device)
                
                out = model(data.x, data.edge_index, data.batch)
                loss = criterion(out.squeeze(), data.y.float())
                test_loss += loss.item() * data.num_graphs
                
                preds = torch.sigmoid(out).squeeze().cpu().numpy()
                labels = data.y.cpu().numpy()
                
                if preds.ndim == 0: preds = [preds]
                if labels.ndim == 0: labels = [labels]
                
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        test_loss /= len(test_dataset)   
        test_auc = roc_auc_score(all_labels, all_preds)

        binary_preds = (np.array(all_preds) > 0.5).astype(int)
        
        test_f1 = f1_score(all_labels, binary_preds)
        test_acc = accuracy_score(all_labels, binary_preds)
        test_prec = precision_score(all_labels, binary_preds, zero_division=0)
        test_rec = recall_score(all_labels, binary_preds, zero_division=0)
        
        metrics = {
            "train_loss": train_loss,
            "test_loss": test_loss,
            "auc": test_auc,
            "f1": test_f1,
            "accuracy": test_acc,
            "precision": test_prec,
            "recall": test_rec
        }
        
        print(f"Epoch {(epoch + 1):02d}/{epochs} - "
              f"Train Loss: {metrics['train_loss']:.4f} | Test Loss: {metrics['test_loss']:.4f} | "
              f"AUC: {metrics['auc']:.4f} | F1: {metrics['f1']:.4f}")
              
        logger.log_epoch(epoch=(epoch + 1), metrics=metrics)

    # Save the trained model inside the logger's unique timestamped folder
    model_path = logger.run_dir / "gnn_model.pth"
    torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    train_gnn()