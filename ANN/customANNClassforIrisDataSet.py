import torch
import torch.nn as nn
import torch.nn.functional as F

class customANNClassForIris(nn.Module):

    def __init__(self, nUnits, nLayers):
        super().__init__()

        self.layers = nn.ModuleDict()
        
        self.nLayers = nLayers

        self.layers['input'] = nn.Linear(4, nUnits)

        for i in range(nLayers):
            self.layers[f'hidden{i}'] = nn.Linear(nUnits, nUnits)

        self.layers['output'] = nn.Linear(nUnits,3)

    def forward(self, x):
        
        x = self.layers['input'](x)

        for i in range (self.nLayers):
            x = F.relu(self.layers[f'hidden{i}'] (x))

        x = self.layers['output'](x)

        return x