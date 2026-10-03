import sys
import os
import random
import copy
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import warnings

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.gan import GraphDiscriminator, GraphGenerator
from src.models.gnn import MFGraph
from src.training.pe_dataset import PEGraphDataset
from src.utils.config import CONFIG
from src.utils.logger import ExperimentLogger

gnn_config = CONFIG["gnn"]
gan_config = CONFIG["gan"]
warnings.filterwarnings("ignore")

def evaluate_loader(loader, model, criterion, device, desc):
    all_preds, all_labels = [], []
    total_loss = 0.0
    
    pbar = tqdm(loader, desc=desc, unit="batch", leave=False)
    with torch.no_grad():
        for batch in pbar:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch).squeeze(-1)
            
            # Compute Validation Loss
            loss = criterion(logits, batch.y.float())
            total_loss += loss.item() * batch.num_graphs
            
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(batch.y.cpu().numpy())
    
    avg_loss = total_loss / len(loader.dataset)
    
    auc = roc_auc_score(all_labels, all_preds)
    bin_preds = (np.array(all_preds) >= 0.5).astype(int)
    acc = accuracy_score(all_labels, bin_preds)
    f1 = f1_score(all_labels, bin_preds)
    prec = precision_score(all_labels, bin_preds, zero_division=0)
    rec = recall_score(all_labels, bin_preds, zero_division=0)
    
    return avg_loss, auc, acc, f1, prec, rec

def train_gan():
    # Force PyTorch to use all available CPU cores for internal operations
    cpu_cores = os.cpu_count() or 6
    torch.set_num_threads(cpu_cores)
    print(f"Allocated {cpu_cores} CPU threads for PyTorch operations.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graphs_dir = project_root / CONFIG["graphs_path"]
    batch_size = gnn_config["batch_size"]
    
    print("\n[Data] Loading January 2018 dataset (Training set)...")
    jan_dataset = PEGraphDataset(graphs_dir / "train")
    
    # 80/20 Split: Train vs. Jan Validation
    train_size = int(0.8 * len(jan_dataset))
    val_size = len(jan_dataset) - train_size
    train_data, val_data = random_split(jan_dataset, [train_size, val_size])

    max_cores = os.cpu_count() or 6
    optimal_workers = min(8, max_cores - 1)
    optimal_workers = max(0, optimal_workers) 
    use_persistent = True
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=optimal_workers, persistent_workers=use_persistent)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=optimal_workers, persistent_workers=use_persistent)
    
    input_dim = jan_dataset[0].x.shape[1]
    noise_dim = 100
    
    print("\n--- [Phase 1/3] Dropout-GAN Training (on 80% Jan split) ---")
    generator = GraphGenerator(noise_dim=noise_dim, output_dim=input_dim).to(device)
    discriminators = [GraphDiscriminator(input_dim=input_dim, hidden_dim=gnn_config["hidden_dim"], k=gnn_config["k"], dropout_rate=gnn_config["dropout_rate"]).to(device) for _ in range(5)]
    
    opt_g = torch.optim.Adam(generator.parameters(), lr=0.002)
    opts_d = [torch.optim.Adam(d.parameters(), lr=0.001) for d in discriminators]
    criterion_gan = torch.nn.BCEWithLogitsLoss()
    gan_epochs = gan_config["epochs"]
    dropout_threshold = gan_config["dropout_rate"]
    
    generator.train()
    for d in discriminators: d.train()
        
    for epoch in range(gan_epochs):
        pbar = tqdm(train_loader, desc=f"[Dropout-GAN] Epoch {epoch+1}/{gan_epochs}", unit="batch")
        for batch in pbar:
            batch = batch.to(device)
            z = torch.randn(batch.num_graphs, noise_dim, device=device)
            fake_x = generator(z)
            
            active_d = []
            for i, disc in enumerate(discriminators):
                if torch.rand(1).item() > dropout_threshold:
                    active_d.append((disc, opts_d[i]))
                    disc.zero_grad()
                    real_pred = disc(batch.x, batch.edge_index, batch.batch)
                    fake_pred = disc(fake_x.detach(), batch.edge_index, batch.batch)
                    loss_d = criterion_gan(real_pred, torch.ones_like(real_pred)) + criterion_gan(fake_pred, torch.zeros_like(fake_pred))
                    loss_d.backward()
                    opts_d[i].step()
                    
            if active_d:
                opt_g.zero_grad()
                loss_g = sum(criterion_gan(d(fake_x, batch.edge_index, batch.batch), torch.ones_like(fake_pred)) for d, _ in active_d) / len(active_d)
                loss_g.backward()
                opt_g.step()
                
                pbar.set_postfix(Loss_G=f"{loss_g.item():.4f}") 
            
        print(f"[Dropout-GAN] Epoch [{epoch+1}/{gan_epochs}] completed.")
        
    print("\n--- [Phase 2/3] Synthetic Data Generation ---")
    generator.eval()
    synthetic_graphs = []
    
    # Generate synthetic graphs equivalent to 50% of your training set size
    num_synthetic = int(0.5 * len(train_data)) 
    
    with torch.no_grad():
        for _ in tqdm(range(num_synthetic), desc="Generating Fake Malware"):
            z = torch.randn(1, noise_dim, device=device)
            random_idx = random.randint(0, len(train_data) - 1)
            base_graph = train_data[random_idx]
            
            fake_features = generator(z).cpu()
            
            if fake_features.shape[0] != base_graph.x.shape[0]:
                fake_features = fake_features.repeat(base_graph.x.shape[0], 1)
                noise = torch.randn_like(fake_features) * 0.15
                fake_features = fake_features + noise
                
            fake_features = torch.nn.functional.normalize(fake_features, p=2.0, dim=-1)

            fake_graph = Data(
                x=fake_features, 
                edge_index=base_graph.edge_index.clone(), 
                y=torch.tensor([1], dtype=torch.long)
            )
            synthetic_graphs.append(fake_graph)
            
    augmented_train_dataset = list(train_data) + synthetic_graphs
    augmented_loader = DataLoader(augmented_train_dataset, batch_size=batch_size, shuffle=True, num_workers=optimal_workers, persistent_workers=use_persistent)
    
    print("\n--- [Phase 3/3] MFGraph Classifier Training ---")
    
    # Initialize ExperimentLogger exactly like GNN, MLP, and LGBM
    logger = ExperimentLogger(experiment_name="gan_training", config=CONFIG)
    exp_dir = logger.run_dir
    
    model = MFGraph(
        input_dim=input_dim, 
        hidden_dim=gnn_config["hidden_dim"], 
        k=gnn_config["k"], 
        dropout_rate=gnn_config["dropout_rate"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    pos_weight = torch.tensor([2.0]).to(device) 
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    best_val_f1 = 0.0
    best_model_weights = None
    
    for epoch in range(1, gnn_config["num_epochs"] + 1):
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(augmented_loader, desc=f"[MFGraph] Epoch {epoch:02d}/{gnn_config['num_epochs']} [Train]", unit="batch")
        
        for batch in train_pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            logits = model(batch.x, batch.edge_index, batch.batch).squeeze(-1)
            logits = torch.clamp(logits, min=-12.0, max=12.0)
            
            loss = criterion(logits, batch.y.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * batch.num_graphs
            train_pbar.set_postfix(Loss=f"{loss.item():.4f}")
            
        avg_train_loss = train_loss / len(augmented_train_dataset)
            
        model.eval()
        val_loss, val_auc, val_acc, val_f1, val_prec, val_rec = evaluate_loader(
            val_loader, model, criterion, device, f"[MFGraph] Epoch {epoch:02d} [Val]"
        )
        
        metrics = {
            "train_loss": avg_train_loss,
            "test_loss": val_loss,
            "auc": val_auc,
            "f1": val_f1,
            "accuracy": val_acc,
            "precision": val_prec,
            "recall": val_rec
        }
        
        print(f"Epoch {epoch:02d}/{gnn_config['num_epochs']} - "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"AUC: {metrics['auc']:.4f} | F1: {metrics['f1']:.4f}")
              
        logger.log_epoch(epoch=epoch, metrics=metrics)
        
        # Track best model based on Validation F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_weights = copy.deepcopy(model.state_dict())
            print(f"  -> New best validation F1 score: {best_val_f1:.4f}")

    # Save the best model at the end of the cycle in the logger directory
    model_path = exp_dir / "model_gan.pth"
    torch.save(best_model_weights, model_path)

    print(f"\n[Success] GAN training completed!")
    print(f"  -> Logs saved to: {exp_dir / 'metrics.csv'}")
    print(f"  -> Model weights saved to: {model_path}")

if __name__ == "__main__":
    train_gan()