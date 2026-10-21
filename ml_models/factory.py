"""One place to create either model by name, using the settings in experiment_config.json.

    build_model("cnn_lstm", 100, config)      -> CNN-LSTM
    build_model("vit_temporal", 100, config)  -> Vision Transformer + temporal Transformer
Both models receive the same dropout / pretrained / frozen settings so the comparison is fair.
"""
from .cnn_lstm import build_model as build_cnn_lstm
from .vit import build_model as build_vit

MODEL_NAMES = ("cnn_lstm", "vit_temporal")


def build_model(name, num_classes, config):
    common = {
        "dropout": config.get("dropout", 0.3),
        "pretrained": config.get("pretrained", True),
        "freeze_backbone": config.get("freeze_backbone", True),
    }
    if name == "cnn_lstm":
        return build_cnn_lstm(
            num_classes=num_classes,
            hidden_size=config.get("lstm_hidden_size", 256),
            num_layers=config.get("lstm_layers", 2),
            **common,
        )
    if name == "vit_temporal":
        return build_vit(
            num_classes=num_classes,
            temporal_heads=config.get("temporal_heads", 8),
            temporal_layers=config.get("temporal_layers", 2),
            **common,
        )
    raise ValueError(f"Unknown model {name!r}; choose one of {MODEL_NAMES}.")
