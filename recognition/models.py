from django.conf import settings
from django.db import models


class RecognitionPrediction(models.Model):
    MODEL_CHOICES = (("cnn_lstm", "CNN-LSTM"), ("vit_temporal", "Vision Transformer"))
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="recognition_predictions", null=True, blank=True)
    model_name = models.CharField(max_length=32, choices=MODEL_CHOICES)
    predicted_sign = models.CharField(max_length=120)
    confidence = models.FloatField()
    latency_ms = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("-created_at",)

    def __str__(self):
        return f"{self.predicted_sign} · {self.model_name} · {self.created_at:%Y-%m-%d}"
