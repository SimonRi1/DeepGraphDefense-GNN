import torch
import torch.nn as nn
from src.models.gnn import MFGraph

class GraphGenerator(nn.Module):
    """
    Generates adversarial node features for the 9 fixed nodes.
    """
    def __init__(self, noise_dim: int, output_dim: int, num_nodes: int = 9):
        super(GraphGenerator, self).__init__()
        self.num_nodes = num_nodes
        self.output_dim = output_dim
        
        # Maps a latent noise vector to a full feature matrix [9, output_dim]
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, num_nodes * output_dim)
        )

    def forward(self, z):
        # z shape: [batch_size, noise_dim]
        out = self.net(z)
        # Reshape to match PyTorch Geometric node feature requirements
        # output shape: [batch_size * 9, output_dim] for batched graphs
        return out.view(-1, self.output_dim)

class GraphDiscriminator(nn.Module):
    """
    Evaluates whether a graph's features are real (benign/untouched) 
    or fake (adversarially generated).
    """
    def __init__(self, input_dim: int, hidden_dim: int, k: int, dropout_rate: float):
        super(GraphDiscriminator, self).__init__()
        # We can reuse the core GNN logic for the discriminator
        self.gnn = MFGraph(input_dim=input_dim, hidden_dim=hidden_dim, k=k, dropout_rate=dropout_rate)

    def forward(self, x, edge_index, batch):
        # Outputs probability of the graph being "real"
        return self.gnn(x, edge_index, batch)