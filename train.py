#!/usr/bin/env python3
import torch
from torchvision import datasets, transforms

import numpy as np

from tqdm import tqdm

# from networks.cnn import CNN
from networks.rnn import CNNLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

transform = transforms.Compose([transforms.ToTensor()])

# Download CIFAR10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

data_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
])

# Create the model
# model = CNN()
model = CNNLSTM()
model.to(device)

EPOCHS = 10

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

progress = tqdm(total=len(train_loader)*EPOCHS, desc="Training")

# Train the model
for epoch in range(EPOCHS):
    for images, labels in train_loader:
        model.train()
        images = images.to(device)
        labels = labels.to(device)

        # perform data augmentation
        images = data_augmentation(images)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress.update(1)

    # currently for every epoch but can adjust to be les frequent
    # calculate train and test loss and accuracy
    if epoch % 1 == 0:
        train_losses = []
        train_accuracies = []

        with torch.no_grad():
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                accuracy = (torch.max(outputs, dim=1)[1] == labels).to(torch.float32).mean()
                train_losses.append(loss.cpu().numpy())
                train_accuracies.append(accuracy.cpu().numpy())

        train_loss, train_accuracy = np.mean(train_losses), np.mean(train_accuracies)


        test_losses = []
        test_accuracies = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                accuracy = (torch.max(outputs, dim=1)[1] == labels).to(torch.float32).mean()
                test_losses.append(loss.cpu().numpy())
                test_accuracies.append(accuracy.cpu().numpy())

        test_loss, test_accuracy = np.mean(test_losses), np.mean(test_accuracies)

        progress.write(f"{('[' + str(epoch + 1) + ']'):8s}   Train: {str(train_accuracy * 100):.6}% ({str(train_loss):.6})   Test: {str(test_accuracy * 100):.6}% ({str(test_loss):.6})")

# Save the model checkpoint
torch.save(model.state_dict(), 'models/cnn.pth')
