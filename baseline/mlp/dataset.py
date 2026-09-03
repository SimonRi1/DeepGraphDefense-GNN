import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path

class EmberFlatDataset(Dataset):
    """
    Dataset loader for the baseline MLP.
    Loads the original EMBER .dat files directly using numpy memmap.
    """
    def __init__(self, data_dir: str, split: str = "train"):
        """
        Args:
            data_dir (str): Path to the directory containing EMBER .dat files.
            split (str): 'train' or 'test' to select the correct subset.
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        
        # EMBER 2018 dataset dimensions
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
            raise FileNotFoundError(f"Missing .dat files in {self.data_dir}. Ensure X_{split}.dat and y_{split}.dat exist.")

        # 1. Read the binary files using numpy memmap
        # mode="c" means copy-on-write. It maps the file without loading it entirely into RAM yet.
        X = np.memmap(x_path, dtype=np.float32, mode="c", shape=(num_samples, FEATURE_DIM))
        y = np.memmap(y_path, dtype=np.float32, mode="c", shape=(num_samples,))
        
        # 2. Filter unlabeled data (y == -1 in EMBER means 'unlabeled')
        mask = y != -1
        
        # 3. Apply the mask and load only the filtered data into memory
        self.X = X[mask]
        self.y = y[mask]
        
        # Quick statistics
        num_benign = (self.y == 0).sum()
        num_malware = (self.y == 1).sum()
        print(f"[{self.split.upper()}] Total labeled samples: {len(self.y)} " f"(Benign: {num_benign}, Malware: {num_malware})")

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.y)

    def __getitem__(self, idx):
        """
        Returns a single sample and its label.
        """
        # Convert the numpy array row to a PyTorch tensor
        features = torch.tensor(self.X[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.float32)
        
        # The label needs an extra dimension [1] to match the neural network's output shape
        label = label.unsqueeze(0) 
        
        return features, label