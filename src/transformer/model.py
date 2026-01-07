# src/transformer/model.py
# -*- coding: utf-8 -*-

"""
Mô đun định nghĩa kiến trúc mô hình Vision Transformer cho bài toán OCR.

Mô hình này là sự kết hợp (Hybrid) giữa:
1. CNN Backbone (VGG-style): Trích xuất đặc trưng hình ảnh từ ảnh đầu vào.
2. Transformer Encoder-Decoder: Học mối quan hệ ngữ cảnh giữa các đặc trưng ảnh
   và giải mã thành chuỗi ký tự.
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    Lớp Positional Encoding (PE) dạng hình sin (Sinusoidal).
    
    Transformer không có khái niệm về thứ tự không gian/thời gian như RNN hay CNN,
    nên ta cần cộng thêm thông tin vị trí vào vector đặc trưng đầu vào.
    
    Công thức:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Khởi tạo Positional Encoding.

        Args:
            d_model (int): Kích thước vector ẩn (embedding dimension).
            dropout (float): Tỷ lệ dropout áp dụng sau khi cộng PE.
            max_len (int): Độ dài chuỗi tối đa được tính toán trước (cache).
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Tạo vector vị trí: [0, 1, 2, ..., max_len-1]
        position = torch.arange(max_len).unsqueeze(1)
        
        # Tính toán div_term (mẫu số trong công thức sin/cos) trong không gian log để ổn định số học
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        # Khởi tạo ma trận PE: (max_len, 1, d_model)
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term) # Các vị trí chẵn dùng Sin
        pe[:, 0, 1::2] = torch.cos(position * div_term) # Các vị trí lẻ dùng Cos
        
        # Đăng ký 'pe' vào buffer để lưu trong state_dict nhưng không được cập nhật bởi optimizer
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass của PE.

        Args:
            x (torch.Tensor): Tensor đầu vào từ bước Embedding hoặc Linear Projection.
                              Shape: (Batch_Size, Seq_Len, Embedding_Dim)
                              (Do model sử dụng batch_first=True)

        Returns:
            torch.Tensor: Tensor đã được cộng thêm thông tin vị trí.
        """
        # Lấy chiều dài chuỗi thực tế từ input (Dimension 1)
        seq_len = x.size(1)
        
        # --- Logic Tự động mở rộng (Dynamic Resizing) ---
        # Nếu chiều dài chuỗi input lớn hơn max_len đã cache, ta cần tính lại PE
        if seq_len > self.pe.size(0):
            new_pe = PositionalEncoding(
                self.dropout.p, 
                self.pe.size(2), 
                max_len=seq_len + 50 # Cộng thêm dư ra một chút
            )
            self.pe = new_pe.pe.to(x.device)
            
        # Cắt PE theo chiều dài chuỗi thực tế -> (Seq_Len, 1, Emb)
        # Permute thành (1, Seq_Len, Emb) để cộng broadcast với x (Batch, Seq, Emb)
        pe_slice = self.pe[:seq_len].permute(1, 0, 2)
        
        # Phép cộng broadcast: (Batch, Seq, Emb) + (1, Seq, Emb)
        x = x + pe_slice
        return self.dropout(x)


class VisionTransformerOCR(nn.Module):
    """
    Mô hình OCR End-to-End sử dụng kiến trúc CNN + Transformer.
    
    Quy trình xử lý:
    Image -> CNN Backbone -> Feature Map -> Flatten -> Transformer Encoder -> Transformer Decoder -> Text
    """

    def __init__(self, num_classes: int, d_model: int = 512, nhead: int = 8, 
                 num_encoder_layers: int = 3, num_decoder_layers: int = 3,
                 dim_feedforward: int = 2048, dropout: float = 0.2):
        """
        Args:
            num_classes (int): Số lượng ký tự trong bộ từ điển (charset) + tokens đặc biệt.
            d_model (int): Kích thước ẩn của Transformer.
            nhead (int): Số lượng head trong Multi-head Attention.
            num_encoder_layers (int): Số lớp Encoder.
            num_decoder_layers (int): Số lớp Decoder.
            dim_feedforward (int): Kích thước lớp ẩn trong Feed Forward Network.
            dropout (float): Tỷ lệ dropout.
        """
        super().__init__()
        self.d_model = d_model

        # --- 1. CNN Backbone (Feature Extractor) ---
        # Sử dụng kiến trúc tương tự VGG để giảm kích thước ảnh và tăng số kênh đặc trưng.
        # Đầu vào: Ảnh xám (1 channel).
        self.cnn_backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            # Block 3 (có BatchNorm)
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), 
            nn.MaxPool2d((2, 1)), # Chỉ giảm chiều cao, giữ nguyên chiều rộng để bảo toàn thông tin chuỗi
            # Block 4
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), 
            nn.MaxPool2d((2, 1)), # Chỉ giảm chiều cao
            # Block 5
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True), 
            nn.MaxPool2d((2, 2)) # Giảm cả 2 chiều lần cuối
        )
        
        # Linear layer để chiếu đặc trưng từ CNN về kích thước d_model của Transformer
        # Input size: 512 * 3 (Do adaptive pooling ép chiều cao về 3 và channels là 512)
        self.input_proj = nn.Linear(512 * 3, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Embedding cho ký tự đích (dùng trong Decoder)
        self.embedding = nn.Embedding(num_classes, d_model)
        
        # --- 2. Transformer ---
        # Quan trọng: batch_first=True giúp input/output có dạng (Batch, Seq, Feature)
        # thay vì mặc định của PyTorch là (Seq, Batch, Feature).
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        # Lớp đầu ra dự đoán xác suất từng ký tự
        self.fc_out = nn.Linear(d_model, num_classes)

    def process_src(self, src: torch.Tensor) -> torch.Tensor:
        """
        Xử lý ảnh đầu vào qua CNN Backbone và chuẩn bị cho Transformer Encoder.
        Đây là bước chuyển đổi từ Không gian Ảnh (2D) sang Không gian Chuỗi (1D).

        Args:
            src (torch.Tensor): Ảnh đầu vào batch. Shape: (Batch, 1, H, W)

        Returns:
            torch.Tensor: Feature sequence. Shape: (Batch, Seq_Len, d_model)
        """
        # 1. Qua CNN Backbone
        src_features = self.cnn_backbone(src)
        bs, c, h, w = src_features.shape
        
        # 2. Adaptive Pooling (Chuẩn hóa chiều cao)
        # Ép chiều cao (Height) về cố định là 3 để đảm bảo kích thước đầu vào Linear Layer luôn đúng
        # bất kể chiều cao ảnh gốc.
        if h != 3: 
            if not hasattr(self, 'adaptive_pool'): 
                self.adaptive_pool = nn.AdaptiveAvgPool2d((3, w))
            src_features = self.adaptive_pool(src_features)
            bs, c, h, w = src_features.shape
        
        # 3. Flatten và Permute (Quan trọng)
        # - view(bs, c * h, w): Gộp kênh (Channel) và chiều cao (Height) lại.
        #   Lúc này mỗi cột pixel (width) được coi là một vector đặc trưng.
        #   Shape: (Batch, Feature_Dim_CNN, Width)
        # - permute(0, 2, 1): Đổi chỗ để đưa Width (đại diện cho chuỗi thời gian) lên trước.
        #   Shape: (Batch, Width, Feature_Dim_CNN) -> Đây chính là (Batch, Seq_Len, Input_Dim)
        src_features = src_features.view(bs, c * h, w).permute(0, 2, 1)
        
        # 4. Projection và PE
        src_features = self.input_proj(src_features) # Chiếu về d_model
        src_features = self.pos_encoder(src_features) # Cộng thông tin vị trí
        
        return src_features

    def forward(
        self, 
        src: torch.Tensor, 
        src_key_padding_mask: Optional[torch.Tensor], 
        tgt: torch.Tensor, 
        tgt_mask: Optional[torch.Tensor], 
        tgt_padding_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass cho quá trình Training.

        Args:
            src: Ảnh đầu vào (Batch, 1, H, W)
            src_key_padding_mask: Mask che phần padding của ảnh (Batch, Seq_Len_Src)
            tgt: Chuỗi đích (Ground Truth) dùng cho Teacher Forcing (Batch, Seq_Len_Tgt)
            tgt_mask: Mask che tương lai (Causal Mask) cho Decoder (Seq_Len_Tgt, Seq_Len_Tgt)
            tgt_padding_mask: Mask che phần padding của chuỗi đích (Batch, Seq_Len_Tgt)

        Returns:
            Logits dự đoán (Batch, Seq_Len_Tgt, Num_Classes)
        """
        # Xử lý ảnh nguồn
        src_processed = self.process_src(src)
        
        # Embedding chuỗi đích + Positional Encoding
        # Scale embedding bằng sqrt(d_model) là kỹ thuật chuẩn trong Transformer ("Attention Is All You Need")
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)
        
        # Transformer Forward
        output = self.transformer(
            src=src_processed, 
            tgt=tgt_emb, 
            tgt_mask=tgt_mask, 
            src_key_padding_mask=src_key_padding_mask, 
            tgt_key_padding_mask=tgt_padding_mask, 
            memory_key_padding_mask=src_key_padding_mask # Mask để Decoder không attend vào padding của ảnh
        )
        
        return self.fc_out(output)

    def encode(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Chỉ chạy phần Encoder (dùng trong quá trình Inference/Prediction).
        
        Returns:
            Memory (Output của Encoder): (Batch, Seq_Len_Src, d_model)
        """
        src_processed = self.process_src(src)
        return self.transformer.encoder(
            src_processed, 
            src_key_padding_mask=src_key_padding_mask
        )

    def decode(
        self, 
        tgt: torch.Tensor, 
        memory: torch.Tensor, 
        tgt_mask: Optional[torch.Tensor], 
        memory_key_padding_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Chỉ chạy phần Decoder (dùng trong vòng lặp autoregressive của Inference).
        
        Args:
            tgt: Chuỗi token đã sinh ra đến thời điểm hiện tại.
            memory: Kết quả từ Encoder (đã tính trước vòng lặp).
        """
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)
        
        return self.transformer.decoder(
            tgt_emb, 
            memory, 
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )