import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils.config import CONFIG
import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx

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
    expected_shape = (9, CONFIG["byte_histogram_dim"])
    
    print("\n--- GRAPH RESULTS ---")
    print(f"Nodes (x) shape: {data.x.shape}")
    print(f"Edges (edge_index) shape: {data.edge_index.shape}")
    print(f"Label (y): {data.y}")

    if data.x.shape == expected_shape:
        print("\n PASS: Padding worked and the graph has the correct dimensions!")
    else:
        print(f"\n ERROR: Expected {expected_shape}, but got {tuple(data.x.shape)}")
    
    print("\nBuilding NetworkX graph...")
    G = builder.to_networkx(data)

    print(f"NetworkX nodes: {G.number_of_nodes()}")
    print(f"NetworkX edges: {G.number_of_edges()}")

    # Visualizza
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    labels = nx.get_node_attributes(G, 'label')

    nx.draw_networkx_nodes(G, pos, node_size=2000,
                        node_color='steelblue', alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, width=2,
                        edge_color='gray', alpha=0.7, ax=ax)
    nx.draw_networkx_labels(G, pos, labels,
                            font_size=9, font_color='white', ax=ax)

    ax.set_title("Feature Graph — topology check", fontsize=14)
    ax.axis('off')
    plt.tight_layout()

    # Salva in experiments/
    output_path = project_root / "experiments" / "feature_graph_topology.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"\nGraph saved in: {output_path}")

if __name__ == "__main__":
    test_graph_construction()