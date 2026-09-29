import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Alias the imports to prevent naming collisions between standard PyTorch and PyTorch Geometric
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Dataset as GeometricDataset

class PEGraphDataset(GeometricDataset):
    """
    Dataset loader for GNN and GAN models.
    Loads pre-processed PyTorch Geometric .pt graph files from disk.
    """
    def __init__(self, root_dir):
        super().__init__()
        print(f"[Dataset] Indexing files in {root_dir}...")
        
        self.file_paths = [Path(root_dir) / f for f in os.listdir(root_dir) if f.endswith('.pt')]
        
        if not self.file_paths:
            raise FileNotFoundError(f"No .pt files found in {root_dir}")
            
        print(f"[Dataset] Successfully indexed {len(self.file_paths)} graph files.")

    def len(self):
        return len(self.file_paths)

    def get(self, idx):
        return torch.load(self.file_paths[idx], weights_only=False)


class EmberFlatDataset(TorchDataset):
    """
    Dataset loader for the baseline MLP (and LGBM).
    Loads the original EMBER .dat files directly using numpy memmap
    and strictly filters training data to Jan 2018 to evaluate concept drift.
    """
    def __init__(self, data_dir: str, split: str = "train"):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split

        FEATURE_DIM = 2381
        TRAIN_SAMPLES = 800000
        TEST_SAMPLES = 200000
        
        print(f"Loading {split} EMBER data directly from {self.data_dir}...")
        
        if self.split == "train":
            x_path = self.data_dir / "X_train.dat"
            y_path = self.data_dir / "y_train.dat"
            num_samples = TRAIN_SAMPLES
        elif self.split == "test":
            x_path = self.data_dir / "X_test.dat"
            y_path = self.data_dir / "y_test.dat"
            num_samples = TEST_SAMPLES
        else:
            raise ValueError("Parameter 'split' must be 'train' or 'test'.")
            
        if not x_path.exists() or not y_path.exists():
            raise FileNotFoundError(f"Missing .dat files in {self.data_dir}")

        X = np.memmap(x_path, dtype=np.float32, mode="c", shape=(num_samples, FEATURE_DIM))
        y = np.memmap(y_path, dtype=np.float32, mode="c", shape=(num_samples,))
        
        valid_label_mask = (y != -1)
        
        if self.split == "train":
            metadata_path = self.data_dir / "metadata.csv"
            if not metadata_path.exists():
                raise FileNotFoundError(f"Missing metadata.csv in {self.data_dir}")
                
            print("[Info] Applying January 2018 date filter to training set...")
            df = pd.read_csv(metadata_path)
            train_df = df[df['subset'] == 'train']
                
            date_mask = train_df['appeared'].str.contains("2018-01", na=False).values
            final_mask = valid_label_mask & date_mask
        else:
            final_mask = valid_label_mask
        
        self.X = X[final_mask]
        self.y = y[final_mask]

        num_benign = (self.y == 0).sum()
        num_malware = (self.y == 1).sum()
        print(f"[{self.split.upper()}] Total labeled samples: {len(self.y)} "
              f"(Benign: {num_benign}, Malware: {num_malware})")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        features = torch.tensor(self.X[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.float32).unsqueeze(0) 
        
        return features, label