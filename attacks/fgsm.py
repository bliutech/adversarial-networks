#!/usr/bin/env python3
# Reference: https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from ..networks.cnn import CNN

def fgsm_attack(image, episilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + episilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

class FGSMTransform:
    """Perform a fast gradient sign attack on an image."""
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __call__(self, image):
        # Set requires_grad attribute of tensor. Important for Attack
        model = CNN().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.load_state_dict(torch.load('../models/cnn.pth'))
        image.requires_grad = True
        # Forward pass the data through the model
        output = model(image)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        # If the initial prediction is wrong, dont bother attacking, just return the initial image
        # if init_pred.item() != target.item():
        #     return image
        # Calculate the loss
        # loss = F.nll_loss(output, target)
        # Zero all existing gradients
        # model.zero_grad()
        # Calculate gradients of model in backward pass
        # loss.backward()
        # Collect datagrad
        data_grad = image.grad.data
        # Call FGSM Attack
        perturbed_image = fgsm_attack(image, self.epsilon, data_grad)
        # Return the perturbed image
        return perturbed_image