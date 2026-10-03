import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from pathlib import Path
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm

import sys
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.config import CONFIG
from src.models.gnn import MFGraph
from baselines.mlp.model import BaselineMLP 

gnn_config = CONFIG["gnn"]
mlp_config = CONFIG["mlp"]

def evaluate_concept_drift(gnn_weights_path, gan_weights_path, mlp_weights_path, lgbm_weights_path, monthly_loaders, device):
    """
    Evaluates 4 models chronologically to analyze concept drift mitigation.
    monthly_loaders: dictionary {2: loader_feb, 3: loader_mar, ..., 12: loader_dec}
    """
    input_dim = CONFIG["max_feature_len"]

    # 1. Initialize and Load All Models
    print("\n[Evaluation] Loading trained models...")
    
    # Baseline GNN
    gnn_model = MFGraph(input_dim, hidden_dim=gnn_config["hidden_dim"], k=gnn_config["k"], dropout_rate=gnn_config["dropout_rate"]).to(device)
    gnn_model.load_state_dict(torch.load(gnn_weights_path))
    gnn_model.eval()
    
    # GAN-Augmented GNN
    gan_model = MFGraph(input_dim, hidden_dim=gnn_config["hidden_dim"], k=gnn_config["k"], dropout_rate=gnn_config["dropout_rate"]).to(device)
    gan_model.load_state_dict(torch.load(gan_weights_path))
    gan_model.eval()

    # MLP Baseline
    mlp_model = BaselineMLP(input_dim, hidden_dims=mlp_config["hidden_dims"], dropout_rate=mlp_config["dropout_rate"]).to(device)
    mlp_model.load_state_dict(torch.load(mlp_weights_path))
    mlp_model.eval()

    # LightGBM Baseline (Loads the booster directly from the .txt file)
    lgbm_model = lgb.Booster(model_file=str(lgbm_weights_path))
    
    results = []
    
    print("\n[Evaluation] Starting chronological testing (Feb - Dec 2018)...")
    with torch.no_grad():
        for month, loader in monthly_loaders.items():
            gnn_preds, gan_preds, mlp_preds, lgbm_preds, labels = [], [], [], [], []
            
            for batch in tqdm(loader, desc=f"Evaluating Month {month:02d}", leave=False, unit="batch"):
                batch = batch.to(device)
                
                # --- Graph Inference ---
                # GNN
                out_gnn = gnn_model(batch.x, batch.edge_index, batch.batch)
                gnn_preds.extend(torch.sigmoid(out_gnn.view(-1)).cpu().numpy())
                
                # GAN
                out_gan = gan_model(batch.x, batch.edge_index, batch.batch)
                gan_preds.extend(torch.sigmoid(out_gan.view(-1)).cpu().numpy())
                
                # --- Tabular Inference ---
                # Collapse graphs into 1D vectors via Mean Pooling for MLP/LGBM
                pooled_features = global_mean_pool(batch.x, batch.batch)
                
                # MLP
                out_mlp = mlp_model(pooled_features)
                mlp_preds.extend(torch.sigmoid(out_mlp.view(-1)).cpu().numpy())
                
                # LGBM (Requires CPU Numpy array)
                lgbm_batch_preds = lgbm_model.predict(pooled_features.cpu().numpy())
                lgbm_preds.extend(lgbm_batch_preds)
                
                labels.extend(batch.y.cpu().numpy())
            
            # Calculate monthly AUC for all models
            gnn_auc = roc_auc_score(labels, gnn_preds)
            gan_auc = roc_auc_score(labels, gan_preds)
            mlp_auc = roc_auc_score(labels, mlp_preds)
            lgbm_auc = roc_auc_score(labels, lgbm_preds)
            
            results.append({
                "Month": month,
                "LGBM_AUC": lgbm_auc,
                "MLP_AUC": mlp_auc,
                "GNN_AUC": gnn_auc,
                "GAN_AUC": gan_auc
            })
            print(f"Month {month:02d} | LGBM: {lgbm_auc:.4f} | MLP: {mlp_auc:.4f} | GNN: {gnn_auc:.4f} | GAN: {gan_auc:.4f}")

    df_results = pd.DataFrame(results)
    
    # 3. Calculate Degradation Rate: (Max - Min) / Max
    def calc_deg(col_name):
        return (df_results[col_name].max() - df_results[col_name].min()) / df_results[col_name].max()

    deg_rates = {
        "LGBM": calc_deg("LGBM_AUC"),
        "MLP": calc_deg("MLP_AUC"),
        "GNN": calc_deg("GNN_AUC"),
        "GAN": calc_deg("GAN_AUC")
    }
    
    print("\n--- Degradation Rate ---")
    print(f"LightGBM:     {deg_rates['LGBM']:.2%}")
    print(f"MLP:          {deg_rates['MLP']:.2%}")
    print(f"Baseline GNN: {deg_rates['GNN']:.2%}")
    print(f"GNN + GAN:    {deg_rates['GAN']:.2%}")
    
    # 4. Save Table to Unified Folder
    out_dir = Path(CONFIG["experiments_path"]) / "drift_comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(out_dir / "monthly_auc_4way_comparison.csv", index=False)
    
    # 5. Generate 4-Way Plot
    plt.figure(figsize=(12, 7))
    
    # Standard Baselines (Dashed Lines)
    plt.plot(df_results["Month"], df_results["LGBM_AUC"], marker='v', linestyle='--', color='orange', alpha=0.8, label=f"LightGBM (Deg: {deg_rates['LGBM']:.1%})")
    plt.plot(df_results["Month"], df_results["MLP_AUC"], marker='^', linestyle='--', color='green', alpha=0.8, label=f"MLP (Deg: {deg_rates['MLP']:.1%})")
    
    # Graph Models (Solid Lines)
    plt.plot(df_results["Month"], df_results["GNN_AUC"], marker='o', linestyle='-', color='red', linewidth=2, label=f"Baseline GNN (Deg: {deg_rates['GNN']:.1%})")
    plt.plot(df_results["Month"], df_results["GAN_AUC"], marker='s', linestyle='-', color='blue', linewidth=2.5, label=f"GNN + GAN (Deg: {deg_rates['GAN']:.1%})")
    
    plt.title('Impact of Concept Drift on Malware Detection (EMBER 2018)', fontsize=14, fontweight='bold')
    plt.xlabel('Evaluation Month (2018)', fontsize=12)
    plt.ylabel('AUC Score', fontsize=12)
    plt.xticks(range(2, 13)) # Months 2 to 12
    plt.ylim(0.5, 1.0)
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower left', fontsize=10, framealpha=0.9)
    plt.tight_layout()
    
    plt.savefig(out_dir / "concept_drift_plot_4way.png", dpi=300)
    print(f"\n[Success] Unified Plot and table saved to {out_dir}")
    plt.show()