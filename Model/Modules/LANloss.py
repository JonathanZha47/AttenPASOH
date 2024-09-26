"""
    This is the module which implements the Loss attention mechanism for the PINN weight assignment.
    However, due to the instability, we have to abandon this module and use other dynamic weight assignment optimization methods.
"""

import torch
import torch.nn as nn
torch.autograd.set_detect_anomaly(True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MultiLayerLinearLAN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3):
        super(MultiLayerLinearLAN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, 1))

    def _init(self):
        for layer in self.net:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_normal_(layer.weight)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x