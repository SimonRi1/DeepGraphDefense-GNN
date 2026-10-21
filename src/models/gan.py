import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from src.models.gnn import MFGraph


def node_positions(batch: torch.Tensor) -> torch.Tensor:
    """Index of every node inside its own graph (PyG batches keep nodes contiguous)."""
    counts = torch.bincount(batch)
    ptr = torch.cumsum(counts, 0) - counts
    return torch.arange(batch.numel(), device=batch.device) - ptr[batch]


class GraphGenerator(nn.Module):
    """
    Generates synthetic node features on top of a real graph topology.

    The topology (edge_index, batch) is borrowed from real malicious graphs, so the
    same generator works for the fixed 9-node PE graphs and the variable NIDS graphs.
    Outputs are squashed with a sigmoid and mapped into the real feature range
    [feat_min, feat_max], so fake samples live in the same space as the real ones.
    """
    def __init__(self, noise_dim: int, output_dim: int, num_nodes: int,
                 feat_min: torch.Tensor, feat_max: torch.Tensor, hidden_dim: int = 128):
        super(GraphGenerator, self).__init__()
        self.noise_dim = noise_dim
        self.num_nodes = num_nodes

        # Node role (PE: feature group, NIDS: flow position in the window)
        self.pos_emb = nn.Embedding(num_nodes, hidden_dim)
        # Graph-level noise (shared by all nodes) + node-level noise
        self.inp = nn.Linear(2 * noise_dim, hidden_dim)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

        # feat_min / feat_max: [num_nodes, output_dim] (per node) or [1, output_dim] (shared)
        self.register_buffer("feat_min", feat_min.float())
        self.register_buffer("feat_range", (feat_max - feat_min).float())

    def forward(self, edge_index, batch):
        num_graphs = int(batch.max().item()) + 1
        pos = node_positions(batch).clamp(max=self.num_nodes - 1)

        z_graph = torch.randn(num_graphs, self.noise_dim, device=batch.device)[batch]
        z_node = torch.randn(batch.numel(), self.noise_dim, device=batch.device)

        h = F.leaky_relu(self.inp(torch.cat([z_graph, z_node], dim=-1)), 0.2) + self.pos_emb(pos)
        h = F.leaky_relu(self.conv1(h, edge_index), 0.2) + h
        h = F.leaky_relu(self.conv2(h, edge_index), 0.2) + h

        idx = pos if self.feat_min.size(0) > 1 else torch.zeros_like(pos)
        return torch.sigmoid(self.out(h)) * self.feat_range[idx] + self.feat_min[idx]


class GraphDiscriminator(nn.Module):
    """
    WGAN critic: outputs an unbounded realness score (no sigmoid).
    Reuses the MFGraph backbone; dropout should be 0 for a stable gradient penalty.
    """
    def __init__(self, input_dim: int, hidden_dim: int, k: int, dropout_rate: float = 0.0):
        super(GraphDiscriminator, self).__init__()
        self.gnn = MFGraph(input_dim=input_dim, hidden_dim=hidden_dim, k=k, dropout_rate=dropout_rate)

    def forward(self, x, edge_index, batch):
        return self.gnn(x, edge_index, batch).squeeze(-1)
