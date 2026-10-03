import sys
import copy
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm
import warnings

# Ensure the project root is in the Python path to allow absolute imports
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.training.pe_dataset import PEGraphDataset
from baselines.mlp.model import BaselineMLP
from src.utils.logger import ExperimentLogger
from src.utils.config import CONFIG

# Extract MLP specific parameters from config
mlp_config = CONFIG["mlp"]
warnings.filterwarnings("ignore")

def convert_graphs_to_tabular(dataset, desc="Converting Graphs"):
    """
    Converte i grafi in vettori 1D facendo la media delle feature dei nodi (Mean Pooling),
    rendendoli compatibili con le reti neurali classiche (MLP).
    """
    X_list, y_list = [], []
    for data in tqdm(dataset, desc=desc):
        graph_vec = data.x.mean(dim=0).numpy()
        X_list.append(graph_vec)
        label = data.y.item() if data.y.numel() == 1 else data.y[0].item()
        y_list.append(label)
    return np.array(X_list), np.array(y_list)

def train_baseline_mlp():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize Logger
    logger = ExperimentLogger(experiment_name="mlp_training", config=CONFIG)
    exp_dir = project_root / CONFIG["experiments_path"] / "mlp_training"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load Data from new monthly split
    graphs_dir = project_root / CONFIG["graphs_path"]
    
    print("\n[Data] Loading January 2018 dataset (Training set)...")
    jan_dataset = PEGraphDataset(graphs_dir / "train")
    
    # 80/20 Split su Gennaio
    train_size = int(0.8 * len(jan_dataset))
    val_size = len(jan_dataset) - train_size
    train_data, val_data = random_split(jan_dataset, [train_size, val_size])

    print("\n[Data] Converting Graph data to Tabular format for MLP...")
    X_train, y_train = convert_graphs_to_tabular(train_data, desc="Train Split")
    X_val, y_val = convert_graphs_to_tabular(val_data, desc="Val Split")

    train_tensor_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_tensor_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

    train_loader = DataLoader(train_tensor_dataset, batch_size=mlp_config["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_tensor_dataset, batch_size=mlp_config["batch_size"], shuffle=False, num_workers=4)

    input_dim = X_train.shape[1]
    print(f"\n[MLP] Detected input dimension: {input_dim}")

    # 3. Initialize Model, Loss, and Optimizer
    model = BaselineMLP(
        input_dim=input_dim,
        hidden_dims=mlp_config["hidden_dims"],
        dropout_rate=mlp_config.get("dropout_rate", 0.5)
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=mlp_config["learning_rate"], weight_decay=1e-4)

    best_val_f1 = 0.0
    best_model_weights = None

    # 4. Training Loop
    print(f"\n[MLP] Starting training loop for {mlp_config['num_epochs']} epochs...")
    for epoch in range(1, mlp_config["num_epochs"] + 1):
        model.train()
        train_loss = 0.0
        
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch:02d} [Train]"):
            features, labels = features.to(device), labels.to(device)
            labels = labels.view(-1)
            
            mask = labels >= 0
            if not mask.any():
                continue
            features = features[mask]
            labels = labels[mask]
            
            optimizer.zero_grad()
            outputs = model(features).view(-1)
            
            outputs = torch.clamp(outputs, min=-12.0, max=12.0)
            loss = criterion(outputs, labels.float())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # 5. Validation Phase (20% Split)
        model.eval()
        test_loss = 0.0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for features, labels in tqdm(val_loader, desc=f"Epoch {epoch:02d} [Val]"):
                features, labels = features.to(device), labels.to(device)
                labels = labels.view(-1)
                
                mask = labels >= 0
                if not mask.any():
                    continue
                features = features[mask]
                labels = labels[mask]
                
                outputs = model(features).view(-1)
                outputs = torch.clamp(outputs, min=-12.0, max=12.0)
                loss = criterion(outputs, labels.float())
                test_loss += loss.item()
                
                probs = torch.sigmoid(outputs)
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        avg_test_loss = test_loss / len(val_loader)
        
        # 6. Calculate Metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        preds_binary = (all_preds > 0.5).astype(int)
        
        metrics = {
            "train_loss": avg_train_loss,
            "test_loss": avg_test_loss, # Mantenuto il nome 'test_loss' per i grafici
            "auc": roc_auc_score(all_labels, all_preds),
            "f1": f1_score(all_labels, preds_binary),
            "accuracy": accuracy_score(all_labels, preds_binary),
            "precision": precision_score(all_labels, preds_binary, zero_division=0),
            "recall": recall_score(all_labels, preds_binary, zero_division=0)
        }
        
        print(f"Epoch {epoch:02d}/{mlp_config['num_epochs']} - "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_test_loss:.4f} | "
              f"AUC: {metrics['auc']:.4f} | F1: {metrics['f1']:.4f}")
        
        logger.log_epoch(epoch=epoch, metrics=metrics)
        
        # Track the best model
        if metrics["f1"] > best_val_f1:
            best_val_f1 = metrics["f1"]
            best_model_weights = copy.deepcopy(model.state_dict())
            print(f"  -> New best validation F1 score: {best_val_f1:.4f}")
        
    # 7. Save final model weights (Best model)
    model_path = exp_dir / "model_mlp.pth"
    torch.save(best_model_weights, model_path)
    
    print(f"\n[Success] MLP Training complete!")
    print(f"  -> Model weights saved in: {model_path}")

if __name__ == "__main__":
    train_baseline_mlp()