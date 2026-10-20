import itertools
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

def build_cicids_graphs(csv_path: str, output_dir: str, window_size: int = 50):
    """
    Parses CIC-IDS2017 CSV, strictly enforces 12 NIDS features, applies log scaling, 
    and builds topology based on Destination Port.
    """
    print(f"\n[CIC-IDS2017] Loading dataset from {csv_path}...")

    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = df.columns.str.strip()

    # Drop Inf and NaN
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Strictly 12 features based precisely on your provided dataset structure
    feature_cols = [
        'Total Length of Fwd Packets',
        'Total Length of Bwd Packets',
        'Fwd Packet Length Mean',
        'Bwd Packet Length Mean',
        'Flow IAT Mean',
        'Fwd IAT Mean',
        'Bwd IAT Mean',
        'Flow Packets/s',
        'Flow Bytes/s',
        'FIN Flag Count',
        'SYN Flag Count',
        'ACK Flag Count',
        'Flow Duration',
        'Total Fwd Packets',
        'Total Backward Packets',
        'Fwd PSH Flags',
    ]

    # Enforce exact feature matching to prevent DataLoader crashes.
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    df['IsMalicious'] = (df['Label'] != 'BENIGN').astype(int)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    num_windows = len(df) // window_size
    print(f"[CIC-IDS2017] Building {num_windows} graphs (Window size: {window_size} flows, Features: 12)...")

    csv_name = Path(csv_path).stem

    for w in tqdm(range(num_windows), desc="Extracting Graphs"):
        window_df = df.iloc[w * window_size : (w + 1) * window_size].reset_index(drop=True)

        # 1. EXTRACT & SCALE NODE FEATURES
        raw_vals = window_df[feature_cols].values.astype(np.float32)
        # Log-scale to compress massive outliers (like Flow Bytes/s being 4,000,000)
        scaled_vals = np.log1p(np.clip(raw_vals, a_min=0.0, a_max=None))
        x = torch.tensor(scaled_vals, dtype=torch.float32)

        # 2. BUILD GRAPH TOPOLOGY (EDGES)
        edges = []

        # Group flows from the same source IP
        if 'Source IP' in window_df.columns:
            for _, group in window_df.groupby('Source IP'):
                indices = group.index.tolist()
                if 1 < len(indices) <= 20:
                    edges.extend(list(itertools.combinations(indices, 2)))
                elif len(indices) > 20:
                    edges.extend(list(zip(indices[:-1], indices[1:])))

        # Group flows with same src-dst IP pair
        if 'Source IP' in window_df.columns and 'Destination IP' in window_df.columns:
            for _, group in window_df.groupby(['Source IP', 'Destination IP']):
                indices = group.index.tolist()
                if 1 < len(indices) <= 10:
                    edges.extend(list(itertools.combinations(indices, 2)))

        if 'Destination Port' in window_df.columns:
            # Group flows targeting the same Destination Port
            for _, group in window_df.groupby('Destination Port'):
                indices = group.index.tolist()
                if 1 < len(indices) <= 20:
                    edges.extend(list(itertools.combinations(indices, 2)))
                elif len(indices) > 20:
                    edges.extend(list(zip(indices[:-1], indices[1:]))) # Chain subset to prevent excessive density

            # Add temporal adjacent links
            temporal_edges = list(zip(range(window_size - 1), range(1, window_size)))
            edges.extend(temporal_edges)
        else:
            # Fallback sequential chain if port is somehow missing
            edges = list(zip(range(window_size - 1), range(1, window_size)))

        if edges:
            edges = list(set(edges)) # Deduplicate
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)  # Undirected
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # 3. GRAPH LABEL
        #graph_label = torch.tensor([1 if window_df['IsMalicious'].sum() > 0 else 0], dtype=torch.long)
        graph_label = torch.tensor([1 if window_df['IsMalicious'].mean() > 0.3 else 0],dtype=torch.long)

        # 4. SAVE
        graph_data = Data(x=x, edge_index=edge_index, y=graph_label)
        torch.save(graph_data, out_path / f"{csv_name}_window_{w:04d}.pt")

    print(f"[Success] Saved {num_windows} graphs to {output_dir}")