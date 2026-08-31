import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


import numpy as np
import torch

from src.graph.feature_graph import FeatureGraphBuilder 

def test_graph_construction():
    # 1. Create dummy tensors with the exact dimensions we expect
    dummy_features = {
        "general": np.random.rand(10).astype(np.float32),
        "header": np.random.rand(10).astype(np.float32),
        "imported": np.random.rand(10).astype(np.float32),
        "exported": np.random.rand(1).astype(np.float32), # The shortest
        "section": np.random.rand(15).astype(np.float32),
        "byte_histogram": np.random.rand(256).astype(np.float32), # The longest
        "byte_entropy": np.random.rand(256).astype(np.float32),
        "data_directories": np.random.rand(32).astype(np.float32),
        "string": np.random.rand(6).astype(np.float32),
    }

    builder = FeatureGraphBuilder()
    
    print("Building the graph...")
    data = builder.build(dummy_features, label=1)

    # 2. Verify that the padding worked
    # We expect 9 nodes, each of length 256 (dimension of byte_histogram)
    expected_shape = (9, 256)
    
    print("\n--- GRAPH RESULTS ---")
    print(f"Nodes (x) shape: {data.x.shape}")
    print(f"Edges (edge_index) shape: {data.edge_index.shape}")
    print(f"Label (y): {data.y}")

    if data.x.shape == expected_shape:
        print("\n PASS: Padding worked and the graph has the correct dimensions!")
    else:
        print(f"\n ERROR: Expected {expected_shape}, but got {tuple(data.x.shape)}")

if __name__ == "__main__":
    test_graph_construction()