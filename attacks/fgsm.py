#!/usr/bin/env python3
# Reference: https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from ..networks.cnn import CNN


class FGSMTransform1:
    """Perform a fast gradient sign attack on an image."""

    def __init__(self, epsilon=0.005):
        self.epsilon = epsilon
        self.model = CNN().to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.load_state_dict(torch.load("./models/cnn_batchsize1.pth"))
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def __call__(self, x):
        # if we process a single image instead of batch, we need to add a fourth dimension
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.to("cuda" if torch.cuda.is_available() else "cpu")
        x.requires_grad = True

        # forward, backward pass to calculate gradient
        output = self.model(x)
        loss = self.criterion(output, labels)
        self.optimizer.zero_grad()
        loss.backward()

        # Collect gradients
        data_grad = x.grad.data

        # Call FGSM Attack, same as fgsm()
        sign_data_grad = data_grad.sign()
        perturbed_images = x + self.epsilon * sign_data_grad
        perturbed_images = torch.clamp(perturbed_images, 0, 1)
        print(perturbed_images.size())
        return perturbed_images
