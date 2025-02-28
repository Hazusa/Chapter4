import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """可学习的位置编码"""

    def __init__(self, d_model, max_len=5):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        return x + self.pos_embed


class LightweightTSClassifier(nn.Module):
    def __init__(self, input_dim=15, num_classes=4, d_model=64, nhead=4, num_layers=4):
        super().__init__()
        self.feature_embed = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, dropout=0.3)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 5, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.feature_embed(x)  # [B,5,15] => [B,5,32]
        x = self.pos_encoder(x)
        x = self.transformer(x)  # [B,5,32]
        x = x.reshape(x.shape[0], -1)
        return self.classifier(x)