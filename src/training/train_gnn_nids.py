import sys
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.gnn import MFGraph
from src.training.nids_dataset import get_nids_splits
from src.utils.logger import ExperimentLogger
from src.utils.config import CONFIG

def train_nids_gnn():
    """
    Trains the NIDS MFGraph on the chronological split of the CIC-IDS-2017 window graphs.
    Class imbalance is handled with pos_weight; the best-validation-AUC weights are
    saved as model_gnn.pth and per-epoch metrics as nids_training_metrics.csv.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[NIDS Training] Using device: {device}")

    # Relative path: run from the project root (python orchestrator.py -t train_nids_gnn)
    data_dir = CONFIG["cic_graphs_path"]

    # Chronological per-scenario split, shared with the baselines and evaluation scripts
    train_loader, val_loader, _ = get_nids_splits(data_dir, batch_size=128, seed=42)
    
    gnn_config = CONFIG["gnn"]
    nids_config = CONFIG["nids"]
    
    all_labels = [data.y.item() for data in train_loader.dataset]
    num_benign = sum(1 for l in all_labels if l == 0)
    num_malicious = sum(1 for l in all_labels if l == 1)
    # Malicious windows are the minority: weight positives by benign/malicious ratio
    pos_weight = torch.tensor([num_benign / num_malicious]).to(device)
    print(f"Class distribution — Benign: {num_benign}, Malicious: {num_malicious}")
    print(f"pos_weight: {pos_weight.item():.4f}")

    input_dim = train_loader.dataset[0].x.shape[1]
    print(f"Detected input dimension: {input_dim}")

    model = MFGraph(
        input_dim=input_dim, 
        hidden_dim=gnn_config["hidden_dim"], 
        k=gnn_config["k"], 
        dropout_rate=gnn_config["dropout_rate"]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=nids_config["learning_rate"], weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Halve the LR when validation AUC stops improving for 2 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
    
    logger = ExperimentLogger(experiment_name="nids_gnn_training", config=CONFIG)
    
    # Tracking AUC to lock in peak performance
    best_auc, best_f1 = -1.0, -1.0
    best_weights = copy.deepcopy(model.state_dict())
    
    # List to store metrics for the CSV
    metrics_history = []

    epochs = gnn_config["num_epochs"]
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch:02d} [Train]"):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Node features are already log1p-scaled by cic_graph.py: no extra scaling here
            logits = model(batch.x, batch.edge_index, batch.batch).squeeze(-1)
            loss = criterion(logits, batch.y.float())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # stabilises early epochs
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        all_preds, all_labels = [], []
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                logits = model(batch.x, batch.edge_index, batch.batch).squeeze(-1)
                val_loss += criterion(logits, batch.y.float()).item()

                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(batch.y.cpu().numpy())

        bin_preds = (np.array(all_preds) >= 0.5).astype(int)
        val_f1 = f1_score(all_labels, bin_preds, zero_division=0)
        val_auc = roc_auc_score(all_labels, all_preds)
        
        epoch_train_loss = train_loss / len(train_loader)
        epoch_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch:02d} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}")
        
        # Append epoch data to history
        metrics_history.append({
            "Epoch": epoch,
            "Train_Loss": epoch_train_loss,
            "Val_Loss": epoch_val_loss,
            "Val_F1": val_f1,
            "Val_AUC": val_auc
        })
        
        scheduler.step(val_auc)
        
        # Save best weights
        # Model selection on validation AUC, F1 breaks ties (AUC often saturates on NIDS)
        if (val_auc, val_f1) > (best_auc, best_f1):
            best_auc, best_f1 = val_auc, val_f1
            best_weights = copy.deepcopy(model.state_dict())
            print(f"  --> [*] New best model saved! (AUC: {best_auc:.4f})")

    # Keep the best-AUC epoch, not the last one
    print(f"\n[Training Complete] Restoring best weights (Peak AUC: {best_auc:.4f})")
    model.load_state_dict(best_weights)

    # Save final model weights
    model_path = logger.run_dir / "model_gnn.pth"
    torch.save(best_weights, model_path)
    
    # Export metrics to CSV
    csv_path = logger.run_dir / "nids_training_metrics.csv"
    df_metrics = pd.DataFrame(metrics_history)
    df_metrics.to_csv(csv_path, index=False)
    
    print(f"\n[Success] NIDS GNN weights saved to: {model_path}")
    print(f"[Success] Training metrics CSV saved to: {csv_path}")

if __name__ == "__main__":
    train_nids_gnn()