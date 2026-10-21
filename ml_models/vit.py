"""Model 2: Vision Transformer + temporal Transformer  (ViT looks at each frame, attention links frames).

In simple words:
  1. A Vision Transformer (ViT-B/16, already trained on ImageNet) cuts each frame into 16x16-pixel
     patches and uses "self-attention" to decide which patches matter. It outputs 768 numbers per frame.
  2. Because ViT only understands single pictures, we add a small *temporal* Transformer on top.
     It looks at all 16 frame-features at once and learns which frames relate to which
     (e.g. "hand starts here, ends there").
  3. A "position" signal is added to every frame first, so the model knows frame order
     (otherwise attention would treat the frames like an unordered bag).
  4. The frames are averaged into one summary and a linear layer gives one score per word.

Shapes used below:  B = clips in a batch, T = frames per clip, C/H/W = colour channels/height/width.
"""
import math

import torch
from torch import nn
from torchvision.models import ViT_B_16_Weights, vit_b_16


class VideoVisionTransformer(nn.Module):
    def __init__(self, num_classes, temporal_heads=8, temporal_layers=2,
                 dropout=0.3, pretrained=True, freeze_backbone=True):
        super().__init__()
        # Load ViT-B/16 and remove its ImageNet classification head so it returns 768 features per frame.
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        frame_model = vit_b_16(weights=weights)
        self.frame_encoder = frame_model
        self.frame_encoder.heads = nn.Identity()
        self.feature_size = 768
        # "Frozen" = ViT's weights stay fixed; only the temporal Transformer and classifier learn.
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for parameter in self.frame_encoder.parameters():
                parameter.requires_grad = False
        # The temporal Transformer: 2 layers of 8-head self-attention across the frames.
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
        # Keep the frozen ViT in evaluation mode while the temporal part trains.
        super().train(mode)
        if self.freeze_backbone:
            self.frame_encoder.eval()
        return self

    def encode_frames(self, video):
        """Step 1: frames -> features, one frame at a time.  [B, T, C, H, W] -> [B, T, 768]"""
        batch, frames = video.shape[:2]
        flattened = video.reshape(batch * frames, *video.shape[2:])
        if self.freeze_backbone:
            with torch.no_grad():
                embeddings = self.frame_encoder(flattened)
        else:
            embeddings = self.frame_encoder(flattened)
        return embeddings.reshape(batch, frames, self.feature_size)

    def forward(self, video):
        """Full model: video clip -> word scores."""
        return self.forward_features(self.encode_frames(video))

    def forward_features(self, sequence):
        """Step 2: features over time -> word scores.  [B, T, 768] -> [B, num_classes]"""
        frames = sequence.shape[1]
        # Add a sine/cosine "position" pattern so each frame carries its place in time
        # (the same trick as the original Transformer paper).
        positions = torch.arange(frames, device=sequence.device, dtype=sequence.dtype).unsqueeze(1)
        dimensions = torch.arange(0, self.feature_size, 2, device=sequence.device, dtype=sequence.dtype)
        scale = torch.exp(dimensions * (-math.log(10000.0) / self.feature_size))
        positional = torch.zeros((frames, self.feature_size), device=sequence.device, dtype=sequence.dtype)
        positional[:, 0::2] = torch.sin(positions * scale)
        positional[:, 1::2] = torch.cos(positions * scale)
        sequence = sequence + positional.unsqueeze(0)
        sequence = self.temporal_encoder(sequence)  # frames "look at" each other with attention
        return self.classifier(self.norm(sequence.mean(dim=1)))  # average over time -> word scores


def build_model(num_classes, **kwargs):
    return VideoVisionTransformer(num_classes=num_classes, **kwargs)
