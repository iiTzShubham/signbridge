"""ViT frame embeddings followed by an explicit temporal Transformer."""
import math

import torch
from torch import nn
from torchvision.models import ViT_B_16_Weights, vit_b_16


class VideoVisionTransformer(nn.Module):
    def __init__(self, num_classes, temporal_heads=8, temporal_layers=2,
                 dropout=0.3, pretrained=True, freeze_backbone=True):
        super().__init__()
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        frame_model = vit_b_16(weights=weights)
        self.frame_encoder = frame_model
        self.frame_encoder.heads = nn.Identity()
        self.feature_size = 768
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for parameter in self.frame_encoder.parameters():
                parameter.requires_grad = False
        layer = nn.TransformerEncoderLayer(
            d_model=self.feature_size,
            nhead=temporal_heads,
            dim_feedforward=self.feature_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            layer, num_layers=temporal_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(self.feature_size)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.feature_size, num_classes))

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.frame_encoder.eval()
        return self

    def encode_frames(self, video):
        # Encode each frame independently: [batch, time, C, H, W] -> [batch, time, feature_size]
        batch, frames = video.shape[:2]
        flattened = video.reshape(batch * frames, *video.shape[2:])
        if self.freeze_backbone:
            with torch.no_grad():
                embeddings = self.frame_encoder(flattened)
        else:
            embeddings = self.frame_encoder(flattened)
        return embeddings.reshape(batch, frames, self.feature_size)

    def forward(self, video):
        return self.forward_features(self.encode_frames(video))

    def forward_features(self, sequence):
        frames = sequence.shape[1]
        # Sinusoidal temporal positions make the transformer sensitive to frame order.
        positions = torch.arange(frames, device=sequence.device, dtype=sequence.dtype).unsqueeze(1)
        dimensions = torch.arange(0, self.feature_size, 2, device=sequence.device, dtype=sequence.dtype)
        scale = torch.exp(dimensions * (-math.log(10000.0) / self.feature_size))
        positional = torch.zeros((frames, self.feature_size), device=sequence.device, dtype=sequence.dtype)
        positional[:, 0::2] = torch.sin(positions * scale)
        positional[:, 1::2] = torch.cos(positions * scale)
        sequence = sequence + positional.unsqueeze(0)
        sequence = self.temporal_encoder(sequence)
        return self.classifier(self.norm(sequence.mean(dim=1)))


def build_model(num_classes, **kwargs):
    return VideoVisionTransformer(num_classes=num_classes, **kwargs)
