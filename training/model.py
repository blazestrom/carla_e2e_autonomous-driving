# training/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PilotNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2),  # out: 24
            nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
        )
        # adaptive pooling avoids manual spatial size calc
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*1*1, 100),
            nn.ELU(),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Linear(10, 1),
            nn.Tanh()  # ensures output in [-1,1]
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.fc(x)
        return x.view(-1)
