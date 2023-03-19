import torch
import torch.nn as nn

# from networks.rnn import CNNLSTM
from networks.cnn import CNN

# model = CNNLSTM()
model = CNN()

model.load_state_dict(torch.load("cnn.pth"))

# Count the number of trainable parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Number of trainable parameters: {num_params}")