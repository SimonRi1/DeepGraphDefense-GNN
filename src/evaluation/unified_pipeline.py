import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, roc_auc_score

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.gnn import MFGraph
from src.training.pe_dataset import PEGraphDataset
from src.training.nids_dataset import get_nids_splits
from src.utils.config import CONFIG

def calculate_impact_mitigation(nids_preds, pe_preds, true_threats):
    """
    Calculates Impact Mitigation based on defense-in-depth layers.
    - Stopped at Network (NIDS) = 100% mitigation (Weight 1.0)
    - Missed by Network, Stopped at Host (PE) = 50% mitigation (Weight 0.5)
    - Missed by both = 0% mitigation
    """
    mitigation_scores = []
    
    for nids_p, pe_p, truth in zip(nids_preds, pe_preds, true_threats):
        if truth == 0:
            continue
            
        if nids_p == 1:
            mitigation_scores.append(1.0)
        elif pe_p == 1:
            mitigation_scores.append(0.5)
        else:
            mitigation_scores.append(0.0)
            
    return np.mean(mitigation_scores) if mitigation_scores else 1.0

def find_latest_checkpoint(search_dir: Path, pattern_name: str):
    """Finds the latest checkpoint matching *.pth or *.pt."""
    checkpoints = list(search_dir.rglob("*.pth")) + list(search_dir.rglob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoint found for '{pattern_name}' in {search_dir}. "
            f"Please verify where the trained {pattern_name} model is stored."
        )
    return max(checkpoints, key=lambda f: f.stat().st_mtime)

def load_unified_system(device):
    """Loads both the PE GNN (input_dim=512) and NIDS GNN (input_dim=9)."""
    gnn_config = CONFIG["gnn"]
    exp_base = project_root / CONFIG.get("experiments_path", "experiments")
    
    # 1. Load PE Model (512-dim node features from PE static analysis)
    pe_dir = exp_base / "gnn_training"
    pe_weights = find_latest_checkpoint(pe_dir, "PE GNN")
    print(f"[Unified Pipeline] Loading PE weights from: {pe_weights}")
    
    pe_model = MFGraph(
        input_dim=512,
        hidden_dim=gnn_config["hidden_dim"], 
        k=gnn_config["k"],
        dropout_rate=gnn_config.get("dropout_rate", 0.5)
    ).to(device)
    pe_model.load_state_dict(torch.load(pe_weights, map_location=device))
    pe_model.eval()
    
    # 2. Load NIDS Model (9-dim flow features from CIC-IDS2017)
    nids_dir = exp_base / "nids_gnn"
    nids_weights = find_latest_checkpoint(nids_dir, "NIDS GNN")
    print(f"[Unified Pipeline] Loading NIDS weights from: {nids_weights}")
    
    nids_model = MFGraph(
        input_dim=9, 
        hidden_dim=gnn_config["hidden_dim"], 
        k=gnn_config["k"],
        dropout_rate=gnn_config.get("dropout_rate", 0.5)
    ).to(device)
    nids_model.load_state_dict(torch.load(nids_weights, map_location=device))
    nids_model.eval()
    
    return pe_model, nids_model

def simulate_mixed_scenarios(nids_loader, pe_loader, num_scenarios=1000):
    """Creates random (NIDS, PE) pairs efficiently using lazy loading."""
    import random
    from tqdm import tqdm
    
    samples_per_type = num_scenarios // 4
    
    nids_0, nids_1 = [], []
    pe_0, pe_1 = [], []
    
    # 1. NIDS dataset è piccolo (es. 570 grafi), possiamo caricarlo tutto con la progress bar
    print("\n[Simulator] Caching NIDS test samples...")
    for data in tqdm(nids_loader.dataset, desc="Scanning NIDS"):
        if data.y.item() == 0:
            nids_0.append(data)
        else:
            nids_1.append(data)
            
    # 2. PE dataset è immenso (750,000 grafi). Usiamo Lazy Loading!
    print(f"[Simulator] Lazy-loading PE samples (Target: {samples_per_type} per class)...")
    pe_indices = list(range(len(pe_loader.dataset)))
    random.seed(42)
    random.shuffle(pe_indices) # Mischiamo gli indici per avere campioni casuali
    
    for idx in tqdm(pe_indices, desc="Scanning PE"):
        data = pe_loader.dataset[idx]
        
        if data.y.item() == 0 and len(pe_0) < samples_per_type:
            pe_0.append(data)
        elif data.y.item() == 1 and len(pe_1) < samples_per_type:
            pe_1.append(data)
            
        # APPENA abbiamo abbastanza campioni, interrompiamo il loop per salvare RAM e Tempo!
        if len(pe_0) >= samples_per_type and len(pe_1) >= samples_per_type:
            break
            
    print(f"\n[Simulator] Extracted -> NIDS (0:{len(nids_0)}, 1:{len(nids_1)}) | PE (0:{len(pe_0)}, 1:{len(pe_1)})")
    
    scenarios = []
    scenario_types = [
        (0, 0), # Normal Traffic + Benign File
        (1, 0), # Attack Traffic + Benign File
        (0, 1), # Normal Traffic + Malware File
        (1, 1)  # Attack Traffic + Malware File
    ]
    
    rng = np.random.default_rng(42)
    
    # Usiamo il random base di Python invece di Numpy per oggetti complessi
    for n_label, p_label in scenario_types:
        n_pool = nids_1 if n_label == 1 else nids_0
        p_pool = pe_1 if p_label == 1 else pe_0
        
        for _ in range(samples_per_type):
            n_sample = random.choice(n_pool)
            p_sample = random.choice(p_pool)
            
            true_threat = 1 if (n_label == 1 or p_label == 1) else 0
            scenarios.append({
                "nids_graph": n_sample,
                "pe_graph": p_sample,
                "true_threat": true_threat,
                "scenario_type": f"NIDS:{n_label}|PE:{p_label}"
            })
            
    return scenarios

def evaluate_unified_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Unified Pipeline] Initializing on {device}...")
    
    pe_model, nids_model = load_unified_system(device)
    
    # 1. Load NIDS holdout test set using your chronological splitter
    nids_dir = project_root / "data" / "graphs" / "nids_graphs"
    _, _, nids_test_loader = get_nids_splits(nids_dir, batch_size=1, seed=42)
    
    # 2. Load PE holdout test set (All Future Months for max concept drift)
    pe_test_path = project_root / "data" / "graphs" / "ember_graphs" / "test"
    
    pe_dataset = PEGraphDataset(str(pe_test_path))
    pe_test_loader = DataLoader(pe_dataset, batch_size=1, shuffle=True)
    
    scenarios = simulate_mixed_scenarios(nids_test_loader, pe_test_loader, num_scenarios=400)
    
    results = []
    
    with torch.no_grad():
        for idx, s in enumerate(tqdm(scenarios, desc="Evaluating Mixed Scenarios")):
            # Process NIDS
            n_batch = s["nids_graph"].to(device)
            n_scaled = torch.log1p(torch.clamp(n_batch.x, min=0.0))
            n_out = nids_model(n_scaled, n_batch.edge_index, None)
            prob_nids = torch.sigmoid(n_out.view(-1)).item()
            
            # Process PE
            p_batch = s["pe_graph"].to(device)
            p_out = pe_model(p_batch.x, p_batch.edge_index, None)
            prob_pe = torch.sigmoid(p_out.view(-1)).item()
            
            # Aggregated Unified Score [0, 1] using Noisy-OR
            unified_score = 1.0 - ((1.0 - prob_nids) * (1.0 - prob_pe))
            
            results.append({
                "Scenario": s["scenario_type"],
                "Prob_NIDS": prob_nids,
                "Prob_PE": prob_pe,
                "Unified_Score": unified_score,
                "Pred_NIDS": int(prob_nids >= 0.5),
                "Pred_PE": int(prob_pe >= 0.5),
                "Pred_Unified": int(unified_score >= 0.5),
                "True_Threat": s["true_threat"]
            })
            
    df = pd.DataFrame(results)
    
    y_true = df["True_Threat"]
    y_pred_unified = df["Pred_Unified"]
    
    auc = roc_auc_score(y_true, df["Unified_Score"])
    f1 = f1_score(y_true, y_pred_unified)
    impact_mitigation = calculate_impact_mitigation(df["Pred_NIDS"], df["Pred_PE"], y_true)
    
    print("\n" + "="*50)
    print(" UNIFIED ALERTING SYSTEM RESULTS")
    print("="*50)
    print(f"Overall Unified AUC:        {auc:.4f}")
    print(f"Overall Unified F1-Score:   {f1:.4f}")
    print(f"Impact Mitigation Metric:   {impact_mitigation:.2%}\n")
    
    print("Breakdown by Scenario Type (Detection Rate):")
    for stype in df["Scenario"].unique():
        sub_df = df[df["Scenario"] == stype]
        if sub_df["True_Threat"].iloc[0] == 1:
            dr = sub_df["Pred_Unified"].mean()
            print(f" - {stype}: {dr:.2%} caught by Unified Pipeline")
        else:
            fpr = sub_df["Pred_Unified"].mean()
            print(f" - {stype} (Benign): {fpr:.2%} False Positive Rate")
            
    out_path = project_root / "experiments" / "unified_pipeline_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n[Success] Detailed unified logs saved to: {out_path}")

if __name__ == "__main__":
    evaluate_unified_pipeline()