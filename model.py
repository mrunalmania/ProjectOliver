import torch
from torch import nn

class NeuralNet(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize) -> None:
        super().__init__()
        self.l1 = nn.Linear(inputSize,hiddenSize)
        self.l2 = nn.Linear(hiddenSize,hiddenSize)
        self.l3 = nn.Linear(hiddenSize,outputSize)
        self.relu = nn.ReLU()
    def forward(self,x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(x)
        out = self.relu(out)
        out = self.l3(out)
        return out
