# src/crnn/model.py
import torch
import torch.nn as nn

class CRNN(nn.Module):
    """
    Định nghĩa model CRNN.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 3)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 3)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.block6 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.pool_v1 = nn.MaxPool2d((3,1), (3,1))
        self.pool_v2 = nn.MaxPool2d((3,1), (3,1))

        self.rnn = nn.LSTM(
            input_size=1024, hidden_size=512, num_layers=2,
            bidirectional=True, batch_first=False, dropout=0.2
        )
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.conv7(x)
        x = self.pool_v1(x)
        x = self.pool_v2(x)

        x = x.squeeze(2)          # (B,1024,T)
        x = x.permute(2,0,1)      # (T,B,1024)
        x, _ = self.rnn(x)        # (T,B,1024)
        x = self.fc(x)            # (T,B,C)
        return x