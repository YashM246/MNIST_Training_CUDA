import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

TRAIN_SIZE = 10000
EPOCHS = 3
LR = 1e-3
BATCH_SIZE = 4
DATA_DIR = "./data"

torch.set_float32_matmul_precision("high")

# Download MNIST Dataset

# Best Practice for MNIST (Boilerplate Code)
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),     # Mean and SD of MNIST
    ]
)

train_dataset = datasets.MNIST(
    root=DATA_DIR, train=True, transform=transform, download=True
)

test_dataset = datasets.MNIST(
    root=DATA_DIR, train=False, transform=transform, download=True
)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Pre-allocate tensors of appropriate size
train_data = torch.zeros(len(train_dataset), 1, 28, 28)
train_labels = torch.zeros(len(train_dataset), dtype=torch.long)
test_data = torch.zeros(len(test_dataset), 1, 28, 28)
test_labels = torch.zeros(len(test_dataset), dtype=torch.long)

# Load training data onto RAM
for idx, (data, label) in enumerate(train_loader):
    start_idx = idx*BATCH_SIZE
    end_idx = start_idx + data.size(0)
    train_data[start_idx: end_idx] = data
    train_labels[start_idx: end_idx] = label

print("Train Data Shape: ", train_data.shape)
print("Train Data Type: ", train_data.dtype)

# Load all test data onto RAM
for idx, (data, label) in enumerate(test_loader):
    start_idx = idx*BATCH_SIZE
    end_idx = start_idx + data.size(0)
    test_data[start_idx: end_idx] = data
    test_labels[start_idx: end_idx] = label

print("Test Data Shape: ", test_data.shape)
print("Test Data Type: ", test_data.dtype)

iters_per_epoch = TRAIN_SIZE // BATCH_SIZE
print("Iters per Epoch: ",iters_per_epoch)

# MLP Class

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, num_classes)
    
    def forward(self, x):
        x = x.reshape(BATCH_SIZE, 28*28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    

# Define Model and Optimizer

model = MLP(in_features=784, hidden_features=256, num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR)

# Train Model

def train(model, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for i in range(iters_per_epoch):
        optimizer.zero_grad()
        data = train_data[i*BATCH_SIZE: (i+1)*BATCH_SIZE]
        target = train_labels[i*BATCH_SIZE: (i+1)*BATCH_SIZE]
        start = time.time()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        end = time.time()
        running_loss += loss.item()
        if i%500 == 0:
            print(f"Epoch: {epoch+1}, Iter: {i}, Loss: {loss}")
            print(f"Iteration Time: {(end-start)*1e3:.4f} sec")
            running_loss = 0

# Eval Function to report avg batch accuracy using the loaded test data

def evaluate(model, test_data, test_labels):
    model.eval()

    total_batch_accuracy = torch.tensor(0.0)
    num_batches = 0

    with torch.no_grad():
        for i in range(len(test_data)//BATCH_SIZE):
            data = test_data[i*BATCH_SIZE: (i+1)*BATCH_SIZE]
            target = test_labels[i*BATCH_SIZE: (i+1)*BATCH_SIZE]
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            correct_batch = (predicted == target).sum().item()
            total_batch = target.size(0)

            if total_batch != 0:
                batch_accuracy = correct_batch / total_batch
                total_batch_accuracy += batch_accuracy
                num_batches += 1
    
    avg_batch_accuracy = total_batch_accuracy / num_batches
    print(f"Average Batch Accuracy: {avg_batch_accuracy*100:.2f}%")

if __name__ == "__main__":
    for epoch in range(EPOCHS):
        train(model, criterion, optimizer, epoch)
        evaluate(model, test_data, test_labels)

    print("Finished Training")