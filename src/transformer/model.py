# src/transformer/model.py
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(0)
        if seq_len > self.pe.size(0): raise ValueError(f"Seq len {seq_len} > PE max_len {self.pe.size(0)}")
        x = x + self.pe[:seq_len];
        return self.dropout(x)

class VisionTransformerOCR(nn.Module):
    def __init__(self, num_classes: int, d_model: int = 512, nhead: int = 8, 
                num_encoder_layers: int = 3, num_decoder_layers: int = 3,
                dim_feedforward: int = 2048, dropout: float = 0.2):
        super().__init__()
        self.d_model = d_model
        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1)),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True), nn.MaxPool2d((2,2))
        )
        
        self.input_proj = nn.Linear(512 * 3, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.embedding = nn.Embedding(num_classes, d_model)
        self.transformer = nn.Transformer(
            d_model, nhead, num_encoder_layers, num_decoder_layers,
            dim_feedforward, dropout, activation='gelu', batch_first=False,
            enable_nested_tensor=False
        )
        self.fc_out = nn.Linear(d_model, num_classes)

    def process_src(self, src: torch.Tensor) -> torch.Tensor:
        src_features = self.cnn_backbone(src)
        bs, c, h, w = src_features.shape
        if h != 3: # Add a check for robustness
            print(f"[WARN] Unexpected CNN output height. Got {h}, expected 3. Applying adaptive pool.");
            if not hasattr(self, 'adaptive_pool'): self.adaptive_pool = nn.AdaptiveAvgPool2d((3, w))
            src_features = self.adaptive_pool(src_features); bs, c, h, w = src_features.shape
        src_features = src_features.view(bs, c * h, w).permute(2, 0, 1)
        src_features = self.input_proj(src_features)
        src_features = self.pos_encoder(src_features)
        return src_features

    def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor, tgt: torch.Tensor, 
                tgt_mask: torch.Tensor, tgt_padding_mask: torch.Tensor) -> torch.Tensor:
        
        src_processed = self.process_src(src)
        
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)

        output = self.transformer(src_processed, tgt_emb,
                                tgt_mask=tgt_mask,
                                src_key_padding_mask=src_key_padding_mask,
                                tgt_key_padding_mask=tgt_padding_mask,
                                memory_key_padding_mask=src_key_padding_mask)
        return self.fc_out(output)

    def encode(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        src_processed = self.process_src(src)
        return self.transformer.encoder(src_processed, src_key_padding_mask=src_key_padding_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor, memory_key_padding_mask: torch.Tensor):
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)
        return self.transformer.decoder(tgt_emb, memory,
                                        tgt_mask=tgt_mask,
                                        memory_key_padding_mask=memory_key_padding_mask)
