# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import time
import numpy as np
import pandas as pd

from collections import Counter


# %% [markdown]
# ### MNIST

# %%
class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


# %%
class MNISTCSVLoader(Dataset):
    def __init__(self, csv_file, transform=None):
        # Read the CSV file using pandas
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        # Return the number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Get the sample at index `idx`
        sample = self.data.iloc[idx]
        
        # The first column is the label
        label = torch.tensor(sample[0], dtype=torch.long)
        
        # The remaining columns are pixel values (convert them to float and normalize)
        image = torch.tensor(sample[1:].values, dtype=torch.float32).view(28, 28)
        
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image.numpy())
        
        return image, label



# %%
# Path to your CSV files
train_csv = './data/mnist/mnist_train.csv'
test_csv = './data/mnist/mnist_test.csv'

# Define transformations (if needed)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Optional, based on your task
])

# Load training and testing data
train_dataset = MNISTCSVLoader(csv_file=train_csv, transform=transform)
test_dataset = MNISTCSVLoader(csv_file=test_csv, transform=transform)

# DataLoader for batching
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


# %%
# Model hyperparameters
input_size = 28 * 28  # 28x28 input pixels flattened
hidden_size = 1024
num_classes = 10  # MNIST has 10 digit classes
learning_rate = 0.001

# Initialize model, loss, optimizer
model = FeedForwardNN(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# %%
def train_model(model, train_loader, criterion, optimizer, num_epochs=2):
    total_step = len(train_loader)
    model.train()
    
    for epoch in range(num_epochs):
        start_time = time.time()
        for i, (images, labels) in enumerate(train_loader):
            # Reshape images to (batch_size, input_size)
            images = images.view(-1, input_size)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')
        
        # Measure epoch time
        end_time = time.time()
        epoch_time = end_time - start_time
        print(f"Epoch {epoch+1} finished in {epoch_time:.2f} seconds")


# %%
def evaluate_model(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.view(-1, input_size)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Test Accuracy of the model: {accuracy:.2f}%')
    return accuracy


# %%
def measure_latency(model, test_loader):
    model.eval()
    inference_times = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, input_size)
            
            # Measure inference time
            start_time = time.time()
            outputs = model(images)
            end_time = time.time()
            inference_times.append(end_time - start_time)
    
    avg_latency = np.mean(inference_times)
    print(f'Average inference latency per batch: {avg_latency:.6f} seconds')
    return avg_latency


# %%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Example usage
param_count = count_parameters(model)
print(f"Total Trainable Parameters: {param_count}")


# %%
def compute_flops(model, input_size):
    flops = 0
    # FLOPs for fc1
    flops += input_size * hidden_size * 2  # Multiply and add operations
    # FLOPs for fc2
    flops += hidden_size * hidden_size * 2
    # FLOPs for fc3
    flops += hidden_size * num_classes * 2
    print(f'Total FLOPs per forward pass: {flops}')
    return flops

# Example usage
compute_flops(model, input_size)


# %%
# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs=2)

# Evaluate the model
evaluate_model(model, test_loader)

# Measure latency
measure_latency(model, test_loader)

# Report parameter count and FLOPs
param_count = count_parameters(model)
flops = compute_flops(model, input_size)

