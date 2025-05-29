import torch
import torch.nn as nn
import torch.nn.functional as F

class customANNClass(nn.Module):

    def __init__(self):
        super().__init__()
        self.input = nn.Linear(2,1)
        self.output = nn.Linear(1,1)

    def forward(self, x):
        x = self.input(x)
        x = F.relu(x)
        x = self.output(x)
        x = torch.sigmoid(x)
        return x