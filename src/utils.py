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
        self.data_dir = data_dir  # Data directory 
        self.transform = transform  # Transformations to apply to images
        self.data = []  # Store image file paths
        self.labels = []  # Store corresponding labels
        # Load images and labels from subfolders
        for subfolder in os.listdir(data_dir):
            subfolder_path = os.path.join(data_dir, subfolder)
            if os.path.isdir(subfolder_path):
                for file_name in os.listdir(subfolder_path):
                    if file_name.endswith('.png'):
                        self.data.append(os.path.join(subfolder_path, file_name))  # Append image path
                        label = int(file_name.split('_')[1].split('.')[0])  # Extract label from filename
                        self.labels.append(label)  # Append corresponding label

    def __len__(self):
        return len(self.data)  # Return number of images in the dataset

    def __getitem__(self, idx):
        img_path = self.data[idx]  # Get image path for given index
        image = Image.open(img_path).convert("L")  # Open image and convert to grayscale
        label = self.labels[idx]  # Get corresponding label
        if self.transform:
            image = self.transform(image)  # Apply transformations 
        return image, label  # Return image and label

def load_femnist(user_id, batch_size=32, data_dir='./src/data/femnist'):
    # Define image transformations for training data
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # Resize images to 28x28 pixels
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean and std deviation
    ])
    client_dir = os.path.join(data_dir, f'client_{user_id}')  # Directory for specific client
    if os.path.exists(client_dir):
        print(f"Loading data for client_{user_id} from {client_dir}...")
        dataset = FEMNISTDataset(client_dir, transform)  # Create dataset instance
        if len(dataset) == 0:
            print(f"Warning: No data found for client_{user_id}.")
            return None
        else:
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # Create DataLoader for training
            return train_loader
    else:
        print(f"Warning: Directory {client_dir} does not exist.")
        return None

def load_test_data(user_id, batch_size=32, data_dir='./src/data/femnist'):
    # Define image transformations for testing data
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # Resize images to 28x28 pixels
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean and std deviation
    ])
    client_dir = os.path.join(data_dir, f'client_{user_id}')  # Directory for the specific client
    test_data = []  # List to store test image paths
    test_labels = []  # List to store corresponding test labels
    # Load test images and labels from subfolders
    for subfolder in range(10):
        subfolder_path = os.path.join(client_dir, str(subfolder))
        if os.path.exists(subfolder_path):
            for file_name in os.listdir(subfolder_path):
                if file_name.endswith('.png'): 
                    img_path = os.path.join(subfolder_path, file_name)  # Get image path
                    test_data.append(img_path)  # Append image path
                    label = int(file_name.split('_')[1].split('.')[0])  # Extract label from filename
                    test_labels.append(label)  # Append corresponding label
    if len(test_data) > 0:
        test_dataset = FEMNISTDataset(client_dir, transform)  # Create dataset for test data
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # Create DataLoader for testing
        return test_loader
    else:
        print(f"Warning: No test data found for client_{user_id}.")
        return None

def train(model, train_loader, epochs=1, lr=0.01, flip_labels=False):
    optimizer = optim.SGD(model.parameters(), lr=lr)  # Define optimizer
    model.train()  # Set model to training mode
    total_loss = 0  # Initialize total loss
    total_correct = 0  # Initialize total correct predictions
    total_samples = 0  # Initialize total samples processed
    for epoch in range(epochs):
        print(f"INFO :      Training epoch {epoch + 1}/{epochs}...")  # Inform about current epoch
        for data, target in train_loader:
            optimizer.zero_grad()  # Clear gradients
            if flip_labels:
                target = (target + 1) % 10  # Flip labels if specified
            output = model(data)  
            loss = F.cross_entropy(output, target)  # Loss
            loss.backward()  
            optimizer.step()  # Update model parameters
            total_loss += loss.item()  # Accumulate loss
            total_samples += target.size(0)  # Update total samples
            pred = output.argmax(dim=1, keepdim=True)  # Get predicted class
            total_correct += pred.eq(target.view_as(pred)).sum().item()  # Count correct predictions
    average_loss = total_loss / len(train_loader)  # Compute average loss
    accuracy = total_correct / total_samples  # Compute accuracy
    return average_loss, accuracy  # Return loss and accuracy 

def evaluate(model, test_loader):
    model.eval()  # Set model to evaluation mode
    test_loss = 0  # Initialize test loss
    correct = 0  # Initialize correct predictions
    with torch.no_grad():  # Disable gradient calculation for evaluation
        for data, target in test_loader:
            output = model(data) 
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # Accumulate test loss
            pred = output.argmax(dim=1, keepdim=True)  # Get predicted class
            correct += pred.eq(target.view_as(pred)).sum().item()  # Count correct predictions
    test_loss /= len(test_loader.dataset)  # Average test loss
    accuracy = correct / len(test_loader.dataset)  # Compute accuracy
    return test_loss, accuracy  # Return loss and accuracy

def save_metrics_to_csv(filename, history):
    # Save training metrics to CSV file
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)  # Create CSV writer object
        writer.writerow(['Round', 'Loss', 'Accuracy'])  # Header
        for round_num, metrics in enumerate(history):
            writer.writerow([round_num + 1, metrics['loss'], metrics['accuracy']])  # Metrics for each round
