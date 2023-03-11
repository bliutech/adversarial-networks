#!/usr/bin/env python3
import torch
from torchvision import datasets, transforms

from tqdm import tqdm

from networks.cnn import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

transform = transforms.Compose([transforms.ToTensor()])

# Download CIFAR10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Create data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Create the model
model = CNN()
model.to(device)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 10

progress = tqdm(total=len(train_loader)*EPOCHS, desc="Training")

# Train the model
for epoch in range(EPOCHS):
    for images, labels in train_loader:
        model.train()
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress.update(1)
    
    progress.write(f'Epoch [{epoch+1}/{10}], Loss: {loss.item():.4f}')

# Save the model checkpoint
torch.save(model.state_dict(), 'model.pth')
