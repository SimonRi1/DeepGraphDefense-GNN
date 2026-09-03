import torch
import torch.nn as nn

class BaselineMLP(nn.Module):
    """
    Standard Multi-Layer Perceptron (MLP) for binary classification.
    Serves as the deep learning baseline for the EMBER dataset.
    """
    def __init__(self, input_dim: int = 2381, hidden_dims: list = [1024, 512, 256], dropout_rate: float = 0.3):
        """
        Args:
            input_dim (int): Number of input features (2381 for EMBER).
            hidden_dims (list): Number of neurons in each hidden layer.
            dropout_rate (float): Dropout probability for regularization.
        """
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim
            
        # Final output layer for binary classification (1 output node)
        layers.append(nn.Linear(current_dim, 1))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.
        Returns:
            torch.Tensor: Raw logits (not probabilities). 
                          Sigmoid is applied later during the loss calculation.
        """
        return self.network(x)