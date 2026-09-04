import sys
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm       # for progress bar to avoid terminal freezing

# Ensure the project root is in the Python path to allow absolute imports
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from baselines.mlp.dataset import EmberFlatDataset
from baselines.mlp.model import BaselineMLP
from src.utils.logger import ExperimentLogger
from src.utils.config import CONFIG

def train_baseline_mlp():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize Logger
    # We pass the entire CONFIG so the logger saves all hyperparameters for reproducibility
    logger = ExperimentLogger(experiment_name="mlp_baseline", config=CONFIG)

    # 2. Load Data
    # Resolves to thesis-project/data/raw/ember2018 using the config path
    data_dir = CONFIG["ember_path"]
    
    train_dataset = EmberFlatDataset(data_dir=str(data_dir), split="train")
    test_dataset = EmberFlatDataset(data_dir=str(data_dir), split="test")

    # Extract MLP specific parameters from config
    mlp_config = CONFIG["mlp"]

    # num_workers=4 speeds up data loading from disk
    train_loader = DataLoader(train_dataset, batch_size=mlp_config["batch_size"], shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=mlp_config["batch_size"], shuffle=False, num_workers=4)

    # 3. Initialize Model, Loss, and Optimizer
    model = BaselineMLP(
        input_dim=mlp_config["input_dim"],
        hidden_dims=mlp_config["hidden_dims"],
        dropout_rate=mlp_config["dropout_rate"]
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=mlp_config["learning_rate"])

    # 4. Training Loop (Progress bar removed, standard python loop used)
    print(f"\nStarting training loop for {mlp_config['num_epochs']} epochs...")
    for epoch in range(1, mlp_config["num_epochs"] + 1):
        model.train()
        train_loss = 0.0
        
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch:02d} [Train]"):
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # 5. Evaluation Phase
        model.eval()
        test_loss = 0.0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for features, labels in tqdm(train_loader, desc=f"Epoch {epoch:02d} [Test]"):
                features, labels = features.to(device), labels.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                # Convert logits to probabilities [0, 1] for metric calculations
                probs = torch.sigmoid(outputs)
                
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        avg_test_loss = test_loss / len(test_loader)
        
        # 6. Calculate Metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        preds_binary = (all_preds > 0.5).astype(int)
        
        metrics = {
            "train_loss": avg_train_loss,
            "test_loss": avg_test_loss,
            "auc": roc_auc_score(all_labels, all_preds),
            "f1": f1_score(all_labels, preds_binary),
            "accuracy": accuracy_score(all_labels, preds_binary),
            "precision": precision_score(all_labels, preds_binary),
            "recall": recall_score(all_labels, preds_binary)
        }
        
        # Print a clean summary string to the terminal
        print(f"Epoch {epoch:02d}/{mlp_config['num_epochs']} - "
              f"Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | "
              f"AUC: {metrics['auc']:.4f} | F1: {metrics['f1']:.4f}")
        
        # Log metrics to our fixed-width file automatically
        logger.log_epoch(epoch=epoch, metrics=metrics)
        
    # 7. Save final model weights
    model_path = logger.run_dir / "mlp_final.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\nTraining complete! Logs and model weights saved in:\n{logger.run_dir}")

if __name__ == "__main__":
    train_baseline_mlp()