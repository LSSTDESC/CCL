import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, hidden_layers, output_dim):
        super().__init__()

        # Input layer
        self.layer_in = nn.Linear(9, hidden_layers[0])

        # Hidden layers
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_layers[i], hidden_layers[i+1]),
                nn.LayerNorm(hidden_layers[i+1]),
                nn.GELU()
            )
            for i in range(len(hidden_layers) - 1)
        ])

        # Output layer
        self.layer_out = nn.Linear(hidden_layers[-1], output_dim)

    def forward(self, x):

        x = F.gelu(self.layer_in(x))

        res = x

        for layer in self.layers:
            x = layer(x)

        x = x + res

        return self.layer_out(x)
