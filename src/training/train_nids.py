import sys
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score
from src.training.nids_dataset import get_nids_splits
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.gnn import MFGraph
from src.utils.logger import ExperimentLogger
from src.utils.config import CONFIG

def train_nids_gnn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[NIDS Training] Using device: {device}")

    data_dir = project_root / "data" / "graphs" / "nids_graphs"
    if not data_dir.exists() or not any(data_dir.iterdir()):
        raise FileNotFoundError(f"No graphs found in {data_dir}. Run the builder first.")

    # REPLACED random_split with the chronological temporal splitter
    train_loader, val_loader, _ = get_nids_splits(data_dir, batch_size=128, seed=42)
    
    input_dim = 9 
    
    gnn_config = CONFIG["gnn"]
    model = MFGraph(
        input_dim=input_dim, 
        hidden_dim=gnn_config["hidden_dim"], 
        k=gnn_config["k"], 
        dropout_rate=gnn_config.get("dropout_rate", 0.5)
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(device))
    
    logger = ExperimentLogger(experiment_name="nids_gnn", config=CONFIG)
    
    # CHANGED: Tracking AUC instead of F1 to avoid getting stuck on fixed thresholds
    best_auc = 0.0
    best_weights = None
    
    # List to store metrics for the CSV
    metrics_history = []

    epochs = gnn_config.get("num_epochs", 20)
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch:02d} [Train]"):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Apply Log Transform to scale down massive network features
            scaled_x = torch.log1p(torch.clamp(batch.x, min=0.0))

            logits = model(scaled_x, batch.edge_index, batch.batch).squeeze(-1)
            loss = criterion(logits, batch.y.float())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                scaled_x = torch.log1p(torch.clamp(batch.x, min=0.0))
                logits = model(scaled_x, batch.edge_index, batch.batch).squeeze(-1)
                
                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(batch.y.cpu().numpy())

        bin_preds = (np.array(all_preds) >= 0.5).astype(int)
        val_f1 = f1_score(all_labels, bin_preds, zero_division=0)
        val_auc = roc_auc_score(all_labels, all_preds)
        
        epoch_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch:02d} | Train Loss: {epoch_train_loss:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}")
        
        # Append epoch data to history
        metrics_history.append({
            "Epoch": epoch,
            "Train_Loss": epoch_train_loss,
            "Val_F1": val_f1,
            "Val_AUC": val_auc
        })
        
        # CHANGED: Saving condition relies on best AUC
        if val_auc > best_auc:
            best_auc = val_auc
            best_weights = copy.deepcopy(model.state_dict())

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