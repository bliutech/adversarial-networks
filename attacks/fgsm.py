#!/usr/bin/env python3
# Reference: https://pytorch.org/tutorials/beginner/fgsm_tutorial.html

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import datasets, transforms
import numpy as np

from networks.cnn import CNN
from networks.rnn import CNNLSTM
    
# Fast gradient sign method
def fgsm(image, epsilon, data_grad):
    """Generate a perturbed image using the Fast Gradient Sign Method."""
    # eta = epsilon * sign(gradient of loss w.r.t input image)
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

class FGSMTransform:
    """Perform a fast gradient sign attack on an image."""

    def __init__(self, epsilon=0.005):
        self.epsilon = epsilon
        self.model = CNN().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.load_state_dict(torch.load("./models/cnn.pth"))
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def __call__(self,sample):
        x, labels = sample
        #if we process a single image instead of batch, we need to add a fourth dimension as the batch dimension
        x = x.unsqueeze(0)
        labels = labels.unsqueeze(0)
        x = x.to(self.device)
        x.requires_grad = True

        # forward, backward pass to calculate gradient
        output = self.model(x)
        loss = self.criterion(output, labels)
        self.model.zero_grad()
        loss.backward()

        # Collect gradients
        data_grad = x.grad.data

        # Call FGSM Attack, same as fgsm()
        sign_data_grad = data_grad.sign()
        perturbed_images = x + self.epsilon * sign_data_grad
        perturbed_images = torch.clamp(perturbed_images, 0, 1)
        return perturbed_images.squeeze()
    
class ToTensor:
    """Convert ndarrays in sample to Tensors. Works the same as transforms.ToTensor() but includes labels"""
    def __call__(self, sample):
        x, label = sample
        return (transforms.functional.to_tensor(x), torch.from_numpy(np.array(label)))


class FGSMTransformCNNLSTM:
    """Perform a fast gradient sign attack on an image."""

    def __init__(self, epsilon=0.005, prob=1):
        self.epsilon = epsilon
        self.model = CNNLSTM().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.load_state_dict(torch.load("./models/cnn-lstm.pth"))
        self.criterion = torch.nn.CrossEntropyLoss()
        self.prob = prob #probability of applying fgsm on an image
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def __call__(self,sample):
        x, labels = sample
        if np.random.rand() >= self.prob:
            return x
        #if we process a single image instead of batch, we need to add a fourth dimension as the batch dimension
        x = x.unsqueeze(0)
        x = x.to(self.device)
        x.requires_grad = True

        # forward, backward pass to calculate gradient
        output = self.model(x)
        loss = self.criterion(output, labels.unsqueeze(0))
        self.model.zero_grad()
        loss.backward()

        # Collect gradients
        data_grad = x.grad.data

        # Call FGSM Attack, same as fgsm()
        sign_data_grad = data_grad.sign()
        perturbed_images = x + self.epsilon * sign_data_grad
        perturbed_images = torch.clamp(perturbed_images, 0, 1)
        return perturbed_images.squeeze(0)