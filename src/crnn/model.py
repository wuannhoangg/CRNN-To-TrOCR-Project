# src/crnn/model.py
"""
Định nghĩa mô hình CRNN (Convolutional Recurrent Neural Network) cho OCR.

Kiến trúc tổng quát:
- CNN backbone trích xuất đặc trưng theo không gian (H, W).
- Pool theo trục dọc (vertical pooling) để giảm chiều cao về 1.
- BiLSTM theo trục thời gian (tương ứng chiều rộng sau CNN) để mô hình hoá chuỗi.
- Linear head ánh xạ đặc trưng -> phân phối lớp ký tự (phục vụ CTC).
"""

import torch
import torch.nn as nn


class CRNN(nn.Module):
    """Mô hình CRNN cho bài toán nhận dạng ký tự theo chuỗi (CTC-based OCR).

    Args:
        num_classes (int): Tổng số lớp đầu ra (bao gồm cả lớp blank cho CTC).
    """

    def __init__(self, num_classes: int):
        super().__init__()

        # Các khối convolution + pooling để giảm kích thước không gian
        # và tăng số kênh đặc trưng.
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 3),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 3),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.block6 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        # Pool theo trục dọc (kernel=(3,1)) để đưa chiều cao về 1,
        # giữ lại trục ngang như trục thời gian (T).
        self.pool_v1 = nn.MaxPool2d((3, 1), (3, 1))
        self.pool_v2 = nn.MaxPool2d((3, 1), (3, 1))

        # BiLSTM xử lý chuỗi theo trục thời gian T (từ đặc trưng CNN).
        self.rnn = nn.LSTM(
            input_size=1024,
            hidden_size=512,
            num_layers=2,
            bidirectional=True,
            batch_first=False,
            dropout=0.2,
        )

        # Linear head: (BiLSTM output dim = 512 * 2 = 1024) -> num_classes.
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        """Lan truyền tiến.

        Args:
            x (torch.Tensor): Tensor ảnh đầu vào shape (B, 1, H, W).

        Returns:
            torch.Tensor: Logits cho CTC shape (T, B, C),
            với T là chiều thời gian sau backbone CNN, C = num_classes.
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.conv7(x)
        x = self.pool_v1(x)
        x = self.pool_v2(x)

        # Sau khi vertical pooling, chiều cao kỳ vọng ~ 1.
        # Chuyển tensor sang dạng chuỗi theo thời gian:
        # (B, 1024, 1, T) -> (B, 1024, T) -> (T, B, 1024).
        x = x.squeeze(2)          # (B,1024,T)
        x = x.permute(2, 0, 1)    # (T,B,1024)

        # BiLSTM: giữ nguyên T, xuất đặc trưng hai chiều (1024).
        x, _ = self.rnn(x)        # (T,B,1024)

        # Linear head để ra logits theo từng timestep.
        x = self.fc(x)            # (T,B,C)
        return x
