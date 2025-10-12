import numpy as np
import torch
import torch.nn as nn

class FNN(nn.Module):
    def __init__(self, layers, activation=nn.GELU(), initialization=nn.init.xavier_uniform_, dtype=torch.float64):
        super(FNN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1], dtype=dtype))
        self.activation = activation
        self.initialization = initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                self.initialization(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.activation(self.layers[i](x))
        x = self.layers[-1](x)
        return x