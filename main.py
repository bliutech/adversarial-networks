#!/usr/bin/env python3
import torch
from torchvision import datasets

# Download CIFAR10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True)