from torch import nn

# Best performing CNN-LSTM hybrid with best CNN, LSTM layers, and FC layer
class CNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )

        self.lstm = nn.LSTM(128, 64, 3, batch_first=True)

        self.fc = nn.Linear(64, 10)
    
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size(0), -1, out.size(1))
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])
        return out

    def size(self):
        return self.network.size()
    
# Older tested archietcture with a shallower network on smaller CNN
class SmallerCNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 64),
        )

        self.lstm = nn.LSTM(128, 64, 3, batch_first=True)

        self.fc = nn.Linear(64, 10)
    
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size(0), -1, out.size(1))
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])
        return out

    def size(self):
        return self.network.size()

# Older tested archietcture with a shallower network and same convolutional sizes towards the end
class SameCNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 64),
        )
    
        self.lstm = nn.LSTM(128, 64, 3, batch_first=True)

        self.fc = nn.Linear(64, 10)
    
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size(0), -1, out.size(1))
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])
        return out

    def size(self):
        return self.network.size()

# Examined better performance of ReLU over ELU
class RELUCNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 64),
        )

        self.lstm = nn.LSTM(128, 64, 3, batch_first=True)

        self.fc = nn.Linear(64, 10)
    
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size(0), -1, out.size(1))
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])
        return out

    def size(self):
        return self.network.size()

# Removed dropout to observe better performance
class DropoutLessCNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(64),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 64),
        )
    
        self.lstm = nn.LSTM(128, 64, 3, batch_first=True)

        self.fc = nn.Linear(64, 10)
    
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size(0), -1, out.size(1))
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])
        return out

    def size(self):
        return self.network.size()

# Ported implementation of the provided TA's CNN network from Keras to Pytorch
class TACNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.5),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 64),
        )
    
        self.lstm = nn.LSTM(128, 64, 3, batch_first=True)

        self.fc = nn.Linear(64, 10)
    
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size(0), -1, out.size(1))
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])
        return out

    def size(self):
        return self.network.size()
