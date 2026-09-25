import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_sort_pool

from pathlib import Path
import sys
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.config import CONFIG

gnn_config = CONFIG["gnn"]

class MalwareGNN(nn.Module):
    """
    DGCNN (Deep Graph Convolutional Neural Network) with SortPooling.
    """
    def __init__(self, input_dim: int, hidden_dim: int = gnn_config["hidden_dim"], k: int = len(CONFIG["feature_names"]), dropout_rate: float = gnn_config["dropout_rate"]):
        """
        Args:
            input_dim (int): Dimension of node features.
            hidden_dim (int): Number of hidden units in GCN layers.
            k (int): The number of nodes to retain in SortPooling (default 9 for your PE graph).
            dropout_rate (float): Dropout probability.
        """
        super(MalwareGNN, self).__init__()
        self.k = k
        
        # 1. Graph Convolutions
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # In DGCNN, the outputs of all GCN layers are concatenated. 
        # So the total feature dimension per node becomes hidden_dim * 3
        total_latent_dim = hidden_dim * gnn_config["num_layers"]
        
        # 2. 1D Convolution (The "Remaining Layer" for the sorted graph)
        # Applies a weighted sum across the k retained nodes to create the final graph embedding
        self.conv1d = nn.Conv1d(
            in_channels=total_latent_dim, 
            out_channels=total_latent_dim, 
            kernel_size=k
        )
        
        # 3. Final Classification Head (Three-Layer Perceptron)
        self.fc1 = nn.Linear(total_latent_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim) # Retained for numerical stability
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        # 1. Message Passing Phase (Multi-Scale)
        # We must save the output of each layer rather than overwriting 'x'
        x1 = F.relu(self.conv1(x, edge_index))
        x2 = F.relu(self.conv2(x1, edge_index))
        x3 = F.relu(self.conv3(x2, edge_index))
        
        # Concatenate outputs from all convolution layers to capture local substructures at multiple scales
        x_cat = torch.cat([x1, x2, x3], dim=-1)
        
        # 2. SortPooling Phase
        # Sorts vertices based on the last feature channel and truncates/pads the graph to exactly 'k' nodes
        # Returns a flattened tensor of shape: (batch_size, k * total_latent_dim)
        x_sorted = global_sort_pool(x_cat, batch, self.k)
        
        # Reshape the flattened tensor into a 1D grid for the convolution layer
        # Shape becomes: (batch_size, total_latent_dim, k)
        x_sorted = x_sorted.view(x_sorted.size(0), self.k, -1).transpose(1, 2)
        
        # 3. 1D Convolution Phase
        # Compress the 'k' ordered nodes into a single graph embedding vector
        x_conv = self.conv1d(x_sorted)      # Output shape: (batch, total_latent_dim, 1)
        x_conv = x_conv.squeeze(-1)         # Flatten to: (batch, total_latent_dim)
        x_conv = F.relu(x_conv)
        x_conv = self.dropout(x_conv)
        
        # 4. Classification Phase (MLP)
        out = self.fc1(x_conv)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        # Return raw logits for BCEWithLogitsLoss
        return out