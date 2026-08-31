import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data

# Topology from Fig 2 of the paper
# Follow the config's file order for the features
# 0="general", 1="header", 2="imported", 3="exported",
# 4="section", 5="byte_histogram", 6="byte_entropy",
# 7="data_directories", 8="string"

EDGE_INDEX = torch.tensor([
    # general - import and export
    [0, 3],
    [0, 2],
    # string - general, section, byte_entropy, byte_histogram, data_directories and header
    [8, 0],
    [8, 4],
    [8, 6],
    [8, 5],
    [8, 7],
    [8, 1]
], dtype=torch.long).t().contiguous()

# bidirectional edges
EDGE_INDEX = torch.cat([
    EDGE_INDEX,
    EDGE_INDEX.flip(0)
], dim=1)

class FeatureGraphBuilder:
    """
    Build the feature graph for a single PE sample.
    Each node = one of the 9 static features.
    Each edge = a semantic relationship between features (fixed topology).
    """
    
    def __init__(self):
        self.edge_index = EDGE_INDEX

    def build(self, features: dict, label: int = -1) -> Data:
        """
        features: a dictionary with keys corresponding to CONFIG["feature_names"]
        label: 0=benign, 1=malignant, -1=unlabelled
        """
        feature_order = [
            "general", "header", "imported", "exported",
            "section", "byte_histogram", "byte_entropy",
            "data_directories", "string"
        ]
        node_features = []
        for name in feature_order:
            vec = features[name]
            node_features.append(torch.tensor(vec, dtype=torch.float32))

        # Uniform-length padding (the longest node determines the dimension)
        max_len = max(v.shape[0] for v in node_features)
        node_features = [
            torch.nn.functional.pad(v, (0, max_len - v.shape[0]))
            for v in node_features
        ]

        x = torch.stack(node_features)  # shape: [9, max_len]

        return Data(
            x=x,
            edge_index=self.edge_index,
            y=torch.tensor([label], dtype=torch.long)
        )

    def to_networkx(self, data: Data, feature_names: list = None) -> nx.Graph:
        """
        Converts to a NetworkX graph for visualisation
        """
        G = nx.Graph()
        names = feature_names or [
            "general", "header", "imported", "exported",
            "section", "byte_hist", "byte_entropy",
            "data_dir", "string"
        ]
        for i, name in enumerate(names):
            G.add_node(i, label=name)

        edges = self.edge_index.t().tolist()
        seen = set()
        for src, dst in edges:
            if (min(src,dst), max(src,dst)) not in seen:
                G.add_edge(src, dst)
                seen.add((min(src,dst), max(src,dst)))

        return G