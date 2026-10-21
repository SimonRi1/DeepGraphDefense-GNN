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
    Chronological 80/10/10 split of the NIDS window graphs, applied separately to
    every capture file (scenario), so each attack type appears in train/val/test
    and test windows are always later in time than training windows.

    Avoids the temporal leakage of a random split: adjacent windows of the same
    capture are almost identical, so a random split would test on near-copies
    of training samples.

    `seed` is unused (the split is deterministic); kept for API compatibility.
    """
    dataset = PEGraphDataset(str(data_dir))
    
    # Sort files to guarantee chronological order (window_0000, window_0001...)
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt')])
    
    # Group windows by capture file (e.g. 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX')
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
    
    # Only training batches are shuffled; val/test keep chronological order
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader