import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.fc1 = nn.Linear(32 * 26 * 26, 10)  

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 32 * 26 * 26)
        x = self.fc1(x)
        return x
