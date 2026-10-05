import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.gnn import MFGraph
from src.utils.config import CONFIG
from src.training.nids_dataset import get_nids_splits

def apply_adversarial_perturbation(batch, feature_idx, multiplier):
    """
    Multiplies a specific feature column by a scalar to simulate evasion tactics.
    """
    perturbed_batch = batch.clone()
    # Explicitly clone the feature matrix to avoid memory reference issues
    perturbed_batch.x = batch.x.clone()
    perturbed_batch.x[:, feature_idx] = perturbed_batch.x[:, feature_idx] * multiplier
    return perturbed_batch

def evaluate_robustness(model, dataloader, device, feature_idx, scale_factors, feature_name="Feature"):
    model.eval()
    results = []
    
    print(f"\n--- Testing Adversarial Robustness: {feature_name} ---")
    
    for scale in scale_factors:
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Scale: {scale}x", leave=False):
                batch = batch.to(device)
                
                # 1. Apply the adversarial multiplier to the RAW features
                if scale != 1.0:
                    batch = apply_adversarial_perturbation(batch, feature_idx, multiplier=scale)
                
                # 2. Scale the features exactly as done in training!
                scaled_x = torch.log1p(torch.clamp(batch.x, min=0.0))
                
                # 3. Inference
                out = model(scaled_x, batch.edge_index, batch.batch)
                probs = torch.sigmoid(out.view(-1)).cpu().numpy()
                
                all_preds.extend(probs)
                all_labels.extend(batch.y.cpu().numpy())
                
        # Calculate metrics
        bin_preds = (np.array(all_preds) >= 0.5).astype(int)
        f1 = f1_score(all_labels, bin_preds, zero_division=0)
        auc = roc_auc_score(all_labels, all_preds)
        
        results.append({
            "Scale": scale,
            "AUC": auc,
            "F1_Score": f1
        })
        print(f"Scale {scale}x | AUC: {auc:.4f} | F1: {f1:.4f}")
        
    df_results = pd.DataFrame(results)
    
    # Calculate Degradation
    baseline_f1 = df_results[df_results['Scale'] == 1.0]['F1_Score'].values[0]
    worst_f1 = df_results['F1_Score'].min()
    
    if baseline_f1 > 0:
        degradation = (baseline_f1 - worst_f1) / baseline_f1
    else:
        degradation = 0.0
        
    print(f"\n[Result] Max F1 Degradation for {feature_name}: {degradation:.2%}")
    return df_results

def run_nids_adversarial():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Adversarial Eval] Using device: {device}")
    
    data_dir = project_root / "data" / "graphs" / "nids_graphs"
    
    # ONLY grab the test_loader (the distant-future chronological split)
    _, _, test_loader = get_nids_splits(data_dir, batch_size=256, seed=42)
    
    # Load Model
    gnn_config = CONFIG["gnn"]
    model = MFGraph(
        input_dim=9, 
        hidden_dim=gnn_config["hidden_dim"], 
        k=gnn_config["k"]
    ).to(device)
    
    # Find the most recently saved weights
    nids_exp_dir = project_root / "experiments" / "nids_gnn"
    gnn_weights = max(nids_exp_dir.rglob("*.pth"), key=lambda f: f.stat().st_mtime)
    model.load_state_dict(torch.load(gnn_weights, map_location=device))
    
    # Evaluate Packet Size
    evaluate_robustness(
        model, test_loader, device, 
        feature_idx=2, 
        scale_factors=[1.0, 0.8, 0.5, 0.1, 2.0, 5.0], 
        feature_name="Packet Size"
    )
    
    # Evaluate Inter-Arrival Time
    evaluate_robustness(
        model, test_loader, device, 
        feature_idx=4, 
        scale_factors=[1.0, 2.0, 5.0, 10.0], 
        feature_name="Inter-Arrival Time"
    )

if __name__ == "__main__":
    run_nids_adversarial()