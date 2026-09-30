import os
import copy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from torch_geometric.loader import DataLoader
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch.utils.data import random_split
import warnings

from pathlib import Path
import sys
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.config import CONFIG
from src.utils.logger import ExperimentLogger
from src.models.gnn import MFGraph
from src.training.pe_dataset import PEGraphDataset

gnn_config = CONFIG["gnn"]
torch.serialization.add_safe_globals([GlobalStorage, DataEdgeAttr, DataTensorAttr])
warnings.filterwarnings("ignore")

def train_gnn():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    logger = ExperimentLogger(experiment_name="gnn_training", config=CONFIG)
    graphs_dir = project_root / CONFIG["graphs_path"]
    
    print("\n[Data] Loading January 2018 dataset...")
    full_jan_dataset = PEGraphDataset(graphs_dir / "train")
    
    # 1. Implement strictly internal 80/20 split on January data
    train_size = int(0.8 * len(full_jan_dataset))
    val_size = len(full_jan_dataset) - train_size
    train_data, val_data = random_split(full_jan_dataset, [train_size, val_size])

    #print("[CPU Optimization] Loading entire datasets into RAM (this takes a moment but speeds up training)...")
    #train_data = [data for data in train_data]
    #val_data = [data for data in val_data]
    
    print("\n[Data] Loading Nov/Dec 2018 Concept Drift Test dataset...")
    test_dataset = PEGraphDataset(graphs_dir / "test")

    max_cores = os.cpu_count() or 4
    optimal_workers = min(8, max_cores - 1)
    optimal_workers = max(0, optimal_workers) 
    use_persistent = True
    
    # 2. Create the three distinct dataloaders
    train_loader = DataLoader(train_data, batch_size=gnn_config["batch_size"], shuffle=True, num_workers=optimal_workers, pin_memory=False, persistent_workers=use_persistent)
    val_loader = DataLoader(val_data, batch_size=gnn_config["batch_size"], shuffle=False, num_workers=optimal_workers, pin_memory=False, persistent_workers=use_persistent)
    test_loader = DataLoader(test_dataset, batch_size=gnn_config["batch_size"], shuffle=False, num_workers=optimal_workers, pin_memory=False, persistent_workers=use_persistent)
    
    sample_graph = full_jan_dataset[0]
    input_dim = sample_graph.x.shape[1]
    print(f"\n[GNN] Detected input dimension: {input_dim}\n")
    
    model = MFGraph(
        input_dim=input_dim, 
        hidden_dim=gnn_config["hidden_dim"],
        k=gnn_config["k"],
        dropout_rate=gnn_config["dropout_rate"]
    ).to(device)
    
    # model = torch.compile(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=gnn_config["learning_rate"], weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    epochs = gnn_config["num_epochs"]
    best_val_f1 = 0.0
    best_model_weights = None

    for epoch in range(epochs):
        # --- Training Phase (80% Jan) ---
        model.train()
        train_loss = 0.0
        
        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out.squeeze(-1), data.y.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.num_graphs
            
        train_loss /= train_size
        
        # --- Validation Phase (20% Jan) ---
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.batch)
                loss = criterion(out.squeeze(-1), data.y.float())
                val_loss += loss.item() * data.num_graphs
                
                preds = torch.sigmoid(out).squeeze(-1).cpu().numpy()
                labels = data.y.cpu().numpy()
                
                # Handle single-item batch shape mismatch
                if preds.ndim == 0: preds = np.array([preds])
                if labels.ndim == 0: labels = np.array([labels])
                
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        val_loss /= val_size   
        val_auc = roc_auc_score(all_labels, all_preds)
        binary_preds = (np.array(all_preds) > 0.5).astype(int)
        
        val_f1 = f1_score(all_labels, binary_preds)
        val_acc = accuracy_score(all_labels, binary_preds)
        val_prec = precision_score(all_labels, binary_preds, zero_division=0)
        val_rec = recall_score(all_labels, binary_preds, zero_division=0)
        
        # Track the best model for concept drift testing
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_weights = copy.deepcopy(model.state_dict())
            print(f"  -> New best validation F1 score: {best_val_f1:.4f}")
        
        metrics = {
            "train_loss": train_loss,
            "test_loss": val_loss, # Kept as 'test_loss' so your logger doesn't break, though it represents validation
            "auc": val_auc,
            "f1": val_f1,
            "accuracy": val_acc,
            "precision": val_prec,
            "recall": val_rec
        }
        
        print(f"Epoch {(epoch + 1):02d}/{epochs} - "
              f"Train Loss: {metrics['train_loss']:.4f} | Val Loss: {metrics['test_loss']:.4f} | "
              f"AUC: {metrics['auc']:.4f} | F1: {metrics['f1']:.4f}")
              
        logger.log_epoch(epoch=(epoch + 1), metrics=metrics)

    # Save the BEST model to disk, not just the last epoch
    model_path = logger.run_dir / "best_gnn_model.pth"
    torch.save(best_model_weights, model_path)
    
    # --- PHASE 3: Concept Drift Test (Nov/Dec 2018) ---
    print("\n--- Final Concept Drift Evaluation (Nov/Dec 2018) ---")
    model.load_state_dict(best_model_weights)
    model.eval()
    
    test_preds, test_labels = [], []
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Evaluating Future Malware"):
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            probs = torch.sigmoid(out).squeeze(-1).cpu().numpy()
            labels = data.y.cpu().numpy()
            
            if probs.ndim == 0: probs = np.array([probs])
            if labels.ndim == 0: labels = np.array([labels])
            
            test_preds.extend(probs)
            test_labels.extend(labels)
            
    test_auc = roc_auc_score(test_labels, test_preds)
    test_bin_preds = (np.array(test_preds) > 0.5).astype(int)
    test_acc = accuracy_score(test_labels, test_bin_preds)
    test_f1 = f1_score(test_labels, test_bin_preds)
    
    print(f"\n[Concept Drift Results] AUC: {test_auc:.4f} | Accuracy: {test_acc:.4f} | F1: {test_f1:.4f}")
    
    # Save test results to a standalone CSV so it doesn't mix with validation epochs
    pd.DataFrame([{
        "auc": test_auc, 
        "accuracy": test_acc, 
        "f1": test_f1
    }]).to_csv(logger.run_dir / "concept_drift_test_metrics.csv", index=False)

if __name__ == "__main__":
    train_gnn()