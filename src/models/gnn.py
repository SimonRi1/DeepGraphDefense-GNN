import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_sort_pool
import warnings

from pathlib import Path
import sys
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.config import CONFIG

gnn_config = CONFIG["gnn"]
warnings.filterwarnings("ignore")

class MFGraph(nn.Module):
    """
    DGCNN (Deep Graph Convolutional Neural Network) with SortPooling.
    """
    def __init__(self, input_dim: int, hidden_dim: int, k: int, dropout_rate: float):
        """
        Args:
            input_dim (int): Dimension of node features.
            hidden_dim (int): Number of hidden units in GCN layers.
            k (int): The number of nodes to retain in SortPooling (default 9 for your PE graph).
            dropout_rate (float): Dropout probability.
        """
        super(MFGraph, self).__init__()
        self.k = k
        
        # Graph Convolutions layers - Convolutional Layers: 3.
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        #self.dropout = nn.Dropout(dropout_rate)
        
        # In DGCNN, the outputs of all GCN layers are concatenated. 
        # So the total feature dimension per node becomes hidden_dim * 3
        total_latent_dim = hidden_dim * gnn_config["num_layers"]
        
        # 2. Graph Representation Learning - Remaining Layer
        # Implements E = f(W * Z^sp) where W is in R^{1 x k}
        # A Linear layer applied across the k dimension precisely achieves this matrix multiplication
        self.w_conv = nn.Linear(self.k, 1)
        
        # 3. Classifier Module (Three-layer MLP)
        mlp_hidden = gnn_config["mlp_hidden_dim"]
        
        # Input layer to first hidden layer
        self.fc1 = nn.Linear(total_latent_dim, mlp_hidden)
        # Second hidden layer
        self.fc2 = nn.Linear(mlp_hidden, mlp_hidden)
        # Output layer
        self.fc3 = nn.Linear(mlp_hidden, 1)
        
        # Dropout applied after every fully connected layer within the hidden layer
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, edge_index, batch):
        # --- Graph Representation Learning ---
        
        # Multi-scale feature aggregation
        z1 = F.relu(self.conv1(x, edge_index))
        z2 = F.relu(self.conv2(z1, edge_index))
        z3 = F.relu(self.conv3(z2, edge_index))
        
        # Concatenate outputs of all h layers
        # Output shape: [num_nodes, total_latent_dim]
        z_concat = torch.cat([z1, z2, z3], dim=-1) 
        
        # Sorts vertices in decreasing order based on the last channel and retains k nodes
        # Output shape: [batch_size, k * total_latent_dim]
        z_sp = global_sort_pool(z_concat, batch, self.k) 
        
        # Reshape and transpose to [batch_size, total_latent_dim, k] for W multiplication
        z_sp = z_sp.view(-1, self.k, z_concat.size(-1)).transpose(1, 2)
        
        # Apply the parametrically represented single-channel Conv 1D layer
        # Output shape: [batch_size, total_latent_dim]
        e_g = F.relu(self.w_conv(z_sp)).squeeze(-1) 
        
        # --- Classifier Module ---
        
        h = F.relu(self.fc1(e_g))
        h = self.dropout(h)
        
        h = F.relu(self.fc2(h))
        h = self.dropout(h)
        
        # Outputs a raw logit compatible with BCEWithLogitsLoss
        out = self.fc3(h) 
        
        return out