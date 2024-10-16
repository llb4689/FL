import torch.nn as nn
import torch.nn.functional as F

# CNN model or FEMNIST
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # Convolutional layer 1: input channels = 1 (grayscale), output channels = 32
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)   
        # Convolutional layer 2: input channels = 32, output channels = 64
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        # Fully connected layer
        self.fc1 = nn.Linear(64 * 7 * 7, 2048)  # Input size for FC layer
        self.fc2 = nn.Linear(2048, num_classes) # Output size is number of classes
        # Xavier initialization for weights
        self._initialize_weights()

    def forward(self, x):
        # First convolution + ReLU + max pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # Second convolution + ReLU + max pooling
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # Flattening the tensor for fully connected layers
        x = x.view(-1, 64 * 7 * 7)
        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # No activation in the final layer as it outputs logits
        return x

    def _initialize_weights(self):
        # Xavier initialization of weights for conv and fc layers
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
