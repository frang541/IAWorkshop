import torch 
from torch import nn

# Define el modelo de red neuronal profunda para regresión
class DeepRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(DeepRegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # Capa totalmente conectada
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Capa de salida

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
