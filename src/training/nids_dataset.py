import pandas as pd
import numpy as np
import os
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from pathlib import Path
from src.training.pe_dataset import PEGraphDataset
from collections import defaultdict

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