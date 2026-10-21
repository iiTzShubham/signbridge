"""ResNet frame features followed by an LSTM temporal classifier."""
import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class CNNLSTM(nn.Module):
    def __init__(self, num_classes, hidden_size=256, num_layers=2, dropout=0.3,
                 pretrained=True, freeze_backbone=True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)
        self.frame_encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_size = 2048
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for parameter in self.frame_encoder.parameters():
                parameter.requires_grad = False
        self.temporal = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.frame_encoder.eval()
        return self

    def encode_frames(self, video):
        # video: [batch, time, channels, height, width] -> [batch, time, feature_size]
        batch, frames = video.shape[:2]
        flattened = video.reshape(batch * frames, *video.shape[2:])
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.frame_encoder(flattened).flatten(1)
        else:
            features = self.frame_encoder(flattened).flatten(1)
        return features.reshape(batch, frames, self.feature_size)

    def forward_features(self, sequence):
        _, (hidden, _) = self.temporal(sequence)
        return self.classifier(self.dropout(hidden[-1]))

    def forward(self, video):
        return self.forward_features(self.encode_frames(video))


def build_model(num_classes, **kwargs):
    return CNNLSTM(num_classes=num_classes, **kwargs)
