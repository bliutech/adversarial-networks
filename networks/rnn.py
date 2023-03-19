from torch import nn

class CNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.LSTM(64, 64, 1, batch_first=True),
        )
    
    def forward(self, x):
        out, _ = self.network(x) # extract the output tensor and the final hidden and cell state tensors from the LSTM layer
        return out
    
    def size(self):
        return self.network.size()