import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from pathlib import Path
from tqdm import tqdm
import itertools
from src.training.pe_dataset import PEGraphDataset
from collections import defaultdict

def build_cicids_graphs(csv_path: str, output_dir: str, window_size: int):
    """
    Parses a CIC-IDS2017 CSV and builds Time-Window graphs.
    If IPs are present, builds Host-Connection graphs. 
    If IPs are missing, falls back to Sequential Temporal graphs.
    """
    print(f"\n[CIC-IDS2017] Loading dataset from {csv_path}...")
    
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    # 9 Features aligned with NIDS literature
    feature_cols = [
        'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 
        'Fwd Packet Length Mean', 'Bwd Packet Length Mean',
        'Flow IAT Mean', 'Fwd IAT Mean', 'Bwd IAT Mean',
        'Flow Packets/s', 'Flow Bytes/s'
    ]
    
    # Check for IP columns
    src_ip_col = 'Source IP' if 'Source IP' in df.columns else ('Src IP' if 'Src IP' in df.columns else None)
    dst_ip_col = 'Destination IP' if 'Destination IP' in df.columns else ('Dst IP' if 'Dst IP' in df.columns else None)
    
    has_ips = src_ip_col and dst_ip_col
    if not has_ips:
        print("[Warning] IP columns not found. Falling back to Sequential Temporal Graphs.")

    df['IsMalicious'] = (df['Label'] != 'BENIGN').astype(int)
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    num_windows = len(df) // window_size
    print(f"[CIC-IDS2017] Building {num_windows} time-window graphs (Window size: {window_size} flows)...")

    csv_name = Path(csv_path).stem # Extracts the name, e.g., 'Monday-WorkingHours'

    for w in tqdm(range(num_windows), desc="Extracting Graphs"):
        window_df = df.iloc[w * window_size : (w + 1) * window_size].reset_index(drop=True)
        
        # 1. Extract Node Features
        x = torch.tensor(window_df[feature_cols].values, dtype=torch.float32)
        
        # 2. Build Edges
        edges = []
        
        if has_ips:
            # Group by Source IP
            for _, group in window_df.groupby(src_ip_col):
                indices = group.index.tolist()
                if len(indices) > 1:
                    edges.extend(list(itertools.combinations(indices, 2)))
                    
            # Group by Destination IP
            for _, group in window_df.groupby(dst_ip_col):
                indices = group.index.tolist()
                if len(indices) > 1:
                    edges.extend(list(itertools.combinations(indices, 2)))
        else:
            # Sequential Temporal Edges: Connect flow N to flow N+1
            src_nodes = np.arange(window_size - 1)
            dst_nodes = np.arange(1, window_size)
            edges = list(zip(src_nodes, dst_nodes))
                
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1) # Make undirected
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            
        # 3. Graph Label
        graph_label = torch.tensor([1 if window_df['IsMalicious'].sum() > 0 else 0], dtype=torch.long)
        
        # 4. Save PyTorch Geometric Data object
        graph_data = Data(x=x, edge_index=edge_index, y=graph_label)
        torch.save(graph_data, out_path / f"{csv_name}_window_{w:04d}.pt")

    print(f"[Success] Saved {num_windows} graphs to {output_dir}")

def get_nids_splits(data_dir: Path, batch_size: int = 128, seed: int = 42):
    """
    Implements Temporal and Spatial bias mitigation.
    Bypasses random shuffling to perform a strict chronological 
    split within each distinct traffic day/scenario.
    """
    dataset = PEGraphDataset(str(data_dir))
    
    # Sort files to guarantee chronological order (window_0000, window_0001...)
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt')])
    
    # Group by Spatial/Scenario Bias (e.g., 'Friday-WorkingHours-Afternoon-DDos')
    scenario_groups = defaultdict(list)
    for idx, f in enumerate(files):
        scenario_name = f.split('_window_')[0]
        scenario_groups[scenario_name].append(idx)
        
    train_indices, val_indices, test_indices = [], [], []
    
    # Apply Temporal Split (80/10/10) chronologically per scenario
    for scenario, indices in scenario_groups.items():
        n = len(indices)
        t_end = int(0.80 * n)
        v_end = int(0.90 * n)
        
        train_indices.extend(indices[:t_end])  # Past
        val_indices.extend(indices[t_end:v_end])  # Near-Future
        test_indices.extend(indices[v_end:])  # Distant-Future
        
    print(f"\n[Dataset] Chronological Split Applied:")
    print(f" - Training (Past):       {len(train_indices)} graphs")
    print(f" - Validation (Present):  {len(val_indices)} graphs")
    print(f" - Test (Future):         {len(test_indices)} graphs")
        
    train_data = Subset(dataset, train_indices)
    val_data = Subset(dataset, val_indices)
    test_data = Subset(dataset, test_indices)
    
    # Shuffle train batches for gradient stability, but maintain chronological integrity for Val/Test
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader