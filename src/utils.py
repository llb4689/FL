import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import csv
from PIL import Image

# FEMNIST Dataset
class FEMNISTDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.data = []
        self.labels = []

        for subfolder in os.listdir(data_dir):
            subfolder_path = os.path.join(data_dir, subfolder)
            if os.path.isdir(subfolder_path):
                for file_name in os.listdir(subfolder_path):
                    if file_name.endswith('.png'):
                        self.data.append(os.path.join(subfolder_path, file_name))
                        label = int(file_name.split('_')[1].split('.')[0])
                        self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = Image.open(img_path).convert("L") 
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def load_femnist(user_id, batch_size=32, data_dir='./src/data/femnist'):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    client_dir = os.path.join(data_dir, f'client_{user_id}')  
    if os.path.exists(client_dir):
        print(f"Loading data for client_{user_id} from {client_dir}...")
        dataset = FEMNISTDataset(client_dir, transform)
        if len(dataset) == 0:
            print(f"Warning: No data found for client_{user_id}.")
            return None
        else:
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            return train_loader
    else:
        print(f"Warning: Directory {client_dir} does not exist.")
        return None

def load_test_data(user_id, batch_size=32, data_dir='./src/data/femnist'):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    client_dir = os.path.join(data_dir, f'client_{user_id}')
    test_data = []
    test_labels = []
    for subfolder in range(10):
        subfolder_path = os.path.join(client_dir, str(subfolder))
        if os.path.exists(subfolder_path):
            for file_name in os.listdir(subfolder_path):
                if file_name.endswith('.png'): 
                    img_path = os.path.join(subfolder_path, file_name)
                    test_data.append(img_path)
                    label = int(file_name.split('_')[1].split('.')[0]) 
                    test_labels.append(label)
    if len(test_data) > 0:
        test_dataset = FEMNISTDataset(client_dir, transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return test_loader
    else:
        print(f"Warning: No test data found for client_{user_id}.")
        return None

def train(model, train_loader, epochs=1, lr=0.01):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    for epoch in range(epochs):
        print(f"INFO :      Training epoch {epoch + 1}/{epochs}...")
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_samples += target.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            total_correct += pred.eq(target.view_as(pred)).sum().item()
    average_loss = total_loss / len(train_loader)
    accuracy = total_correct / total_samples
    return average_loss, accuracy 

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    return test_loss, accuracy

def save_metrics_to_csv(filename, history):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Round', 'Loss', 'Accuracy'])
        for round_num, metrics in enumerate(history):
            writer.writerow([round_num+1, metrics['loss'], metrics['accuracy']])

def plot_metrics(history):
    """Plot training loss and accuracy over rounds."""
    rounds = range(len(history))
    losses = [metrics['loss'] for metrics in history]
    accuracies = [metrics['accuracy'] for metrics in history]  
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(rounds, losses, label="Loss")
    plt.xlabel("Rounds")
    plt.ylabel("Loss")
    plt.title("Loss over Rounds")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(rounds, accuracies, label="Accuracy", color="green")
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Rounds")
    plt.legend()
    plt.tight_layout()
    plt.show()