"""Model 1: CNN-LSTM  (ResNet-50 looks at each frame, an LSTM reads the frames in order).

In simple words:
  1. A video clip is a short list of pictures (frames), e.g. 16 frames.
  2. ResNet-50 (a CNN already trained on millions of ImageNet photos) turns each frame into a
     list of 2048 numbers that describe what is in the picture ("features").
  3. An LSTM reads those 16 feature lists one after another and remembers how things change
     over time - that change is the movement of the sign.
  4. A final linear layer turns the LSTM's summary into one score per word (100 words).
     The word with the highest score is the prediction.

Shapes used below:  B = clips in a batch, T = frames per clip, C/H/W = colour channels/height/width.
"""
import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class CNNLSTM(nn.Module):
    def __init__(self, num_classes, hidden_size=256, num_layers=2, dropout=0.3,
                 pretrained=True, freeze_backbone=True):
        super().__init__()
        # Load ResNet-50 with ImageNet weights and drop its last layer (the 1000 ImageNet classes),
        # so it outputs a 2048-number feature vector per frame instead.
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)
        self.frame_encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_size = 2048
        # "Frozen" = ResNet's weights are not changed during training; only the LSTM learns.
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for parameter in self.frame_encoder.parameters():
                parameter.requires_grad = False
        # The LSTM reads the sequence of frame features and keeps a "memory" (hidden state).
        self.temporal = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)  # randomly hides some numbers while training to reduce overfitting
        self.classifier = nn.Linear(hidden_size, num_classes)  # memory -> one score per word

    def train(self, mode=True):
        # Keep the frozen ResNet in evaluation mode even while the LSTM trains, so its
        # BatchNorm statistics never change.
        super().train(mode)
        if self.freeze_backbone:
            self.frame_encoder.eval()
        return self

    def encode_frames(self, video):
        """Step 1: frames -> features.  [B, T, C, H, W] -> [B, T, 2048]"""
        batch, frames = video.shape[:2]
        flattened = video.reshape(batch * frames, *video.shape[2:])  # treat every frame as its own picture
        if self.freeze_backbone:
            with torch.no_grad():  # no gradients needed for a frozen network (faster, less memory)
                features = self.frame_encoder(flattened).flatten(1)
        else:
            features = self.frame_encoder(flattened).flatten(1)
        return features.reshape(batch, frames, self.feature_size)  # put frames back into clips

    def forward_features(self, sequence):
        """Step 2: features over time -> word scores.  [B, T, 2048] -> [B, num_classes]"""
        _, (hidden, _) = self.temporal(sequence)
        # hidden[-1] is the top LSTM layer's memory after it has seen the last frame.
        return self.classifier(self.dropout(hidden[-1]))

    def forward(self, video):
        """Full model: video clip -> word scores."""
        return self.forward_features(self.encode_frames(video))


def build_model(num_classes, **kwargs):
    return CNNLSTM(num_classes=num_classes, **kwargs)
