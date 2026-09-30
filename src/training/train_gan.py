import sys
import os
import random
from datetime import datetime
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
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

gnn_config = CONFIG["gnn"]
gan_config = CONFIG["gan"]
warnings.filterwarnings("ignore")

def evaluate_loader(loader, model, device, desc):
    all_preds, all_labels = [], []
    pbar = tqdm(loader, desc=desc, unit="batch", leave=False)
    with torch.no_grad():
        for batch in pbar:
            batch = batch.to(device)
            # The function now safely uses the passed model and device
            probs = torch.sigmoid(model(batch.x, batch.edge_index, batch.batch).squeeze(-1)).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(batch.y.cpu().numpy())
    
    auc = roc_auc_score(all_labels, all_preds)
    bin_preds = [1 if p >= 0.5 else 0 for p in all_preds]
    acc = accuracy_score(all_labels, bin_preds)
    f1 = f1_score(all_labels, bin_preds)
    return auc, acc, f1

def train_gan():
    # Force PyTorch to use all available CPU cores for internal operations
    cpu_cores = os.cpu_count() or 4
    torch.set_num_threads(cpu_cores)
    print(f"Allocated {cpu_cores} CPU threads for PyTorch operations.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graphs_dir = project_root / CONFIG["graphs_path"]
    batch_size = gnn_config["batch_size"]
    
    print("\n[Data] Loading datasets...")
    jan_dataset = PEGraphDataset(graphs_dir / "train")
    test_dataset = PEGraphDataset(graphs_dir / "test")
    
    # 80/20 Split: Train vs. Jan Validation/Test
    train_size = int(0.8 * len(jan_dataset))
    val_size = len(jan_dataset) - train_size
    train_data, val_data = random_split(jan_dataset, [train_size, val_size])

    max_cores = os.cpu_count() or 4
    optimal_workers = min(8, max_cores - 1)
    optimal_workers = max(0, optimal_workers) 
    use_persistent = True
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=optimal_workers, persistent_workers=use_persistent)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=optimal_workers, persistent_workers=use_persistent)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=optimal_workers, persistent_workers=use_persistent)
    
    input_dim = jan_dataset[0].x.shape[1]
    noise_dim = 100
    
    print("\n--- [Phase 1/3] Dropout-GAN Training (on 80% Jan split) ---")
    generator = GraphGenerator(noise_dim=noise_dim, output_dim=input_dim).to(device)
    discriminators = [GraphDiscriminator(input_dim=input_dim, hidden_dim=gnn_config["hidden_dim"], k=gnn_config["k"], dropout_rate=gnn_config["dropout_rate"]).to(device) for _ in range(5)]
    
    # Keep Generator LR at 0.002
    opt_g = torch.optim.Adam(generator.parameters(), lr=0.002)

    # Lower Discriminator LR to 0.001 so they don't overpower the generator
    opts_d = [torch.optim.Adam(d.parameters(), lr=0.001) for d in discriminators]
    criterion_gan = torch.nn.BCEWithLogitsLoss()
    gan_epochs = gan_config["epochs"]
    dropout_threshold = gan_config["dropout_rate"]  # Probability of dropping a discriminator in each batch
    
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
    
    # Generate synthetic graphs equivalent to 20% of your training set size
    num_synthetic = int(0.2 * len(train_data)) 
    
    with torch.no_grad():
        for _ in tqdm(range(num_synthetic), desc="Generating Fake Malware"):
            z = torch.randn(1, noise_dim, device=device)
            
            # Borrow a valid topological structure from a random training graph
            random_idx = random.randint(0, len(train_data) - 1)
            base_graph = train_data[random_idx]
            
            # Ensure the fake features match the node count expected by the edge_index
            fake_features = generator(z).cpu()
            
            # If your generator outputs a 1D graph embedding instead of node features, expand it
            if fake_features.shape[0] != base_graph.x.shape[0]:
                fake_features = fake_features.repeat(base_graph.x.shape[0], 1)

                noise = torch.randn_like(fake_features) * 0.05
                fake_features = fake_features + noise
                
            # L2 Normalize the synthetic features so they perfectly match the scale of your real dataset   
            fake_features = torch.nn.functional.normalize(fake_features, p=2.0, dim=-1)

            fake_graph = Data(
                x=fake_features, 
                edge_index=base_graph.edge_index.clone(), 
                y=torch.tensor([1], dtype=torch.long)
            )
            synthetic_graphs.append(fake_graph)
            
    augmented_train_dataset = list(train_data) + synthetic_graphs
    # Shuffle well so the synthetic data is distributed randomly in the batches
    augmented_loader = DataLoader(augmented_train_dataset, batch_size=batch_size, shuffle=True, num_workers=optimal_workers, persistent_workers=use_persistent)
    
    print("\n--- [Phase 3/3] MFGraph Classifier Training ---")
    model = MFGraph(
        input_dim=input_dim, 
        hidden_dim=gnn_config["hidden_dim"], 
        k=gnn_config["k"], 
        dropout_rate=gnn_config["dropout_rate"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # Calculate a rough weight for the positive class to balance the loss function
    pos_weight = torch.tensor([0.8]).to(device) 
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = project_root / CONFIG["experiments_path"] / "gan_training" / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_log = []
    
    for epoch in range(1, gnn_config["num_epochs"] + 1):
        model.train()
        train_pbar = tqdm(augmented_loader, desc=f"[MFGraph] Epoch {epoch:02d}/{gnn_config['num_epochs']} [Train]", unit="batch")
        for batch in train_pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Extract logits and clamp them to prevent BCE explosion
            logits = model(batch.x, batch.edge_index, batch.batch).squeeze(-1)
            logits = torch.clamp(logits, min=-12.0, max=12.0)
            
            loss = criterion(logits, batch.y.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_pbar.set_postfix(Loss=f"{loss.item():.4f}")
            
        # Validation strictly on the 20% January holdout
        model.eval()
        
        # Evaluate Validation ONLY during the epoch loop (Jan 20%)
        val_auc, val_acc, val_f1 = evaluate_loader(val_loader, model, device, f"[MFGraph] Epoch {epoch:02d} [Val]")
        
        print(f"Epoch {epoch:02d} | Val AUC: {val_auc:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        
        metrics_log.append({
            "epoch": epoch, 
            "val_auc": val_auc, "val_accuracy": val_acc, "val_f1": val_f1
        })
        
    print("\n--- Final Test Set Evaluation (Nov-Dec) ---")
    model.eval()
    test_auc, test_acc, test_f1 = evaluate_loader(test_loader, model, device, "[MFGraph] Final Test")
    print(f"Final Test AUC: {test_auc:.4f} | Acc: {test_acc:.4f} | F1: {test_f1:.4f}\n")
    
    # Append final test metrics to the dataframe before saving
    df_metrics = pd.DataFrame(metrics_log)
    df_metrics["final_test_auc"] = test_auc
    df_metrics["final_test_accuracy"] = test_acc
    df_metrics["final_test_f1"] = test_f1
    
    df_metrics.to_csv(exp_dir / "metrics.csv", index=False)
    print(f"\n[Success] GAN training completed! Metrics saved to: {exp_dir / 'metrics.csv'}")

if __name__ == "__main__":
    train_gan()