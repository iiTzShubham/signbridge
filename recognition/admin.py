from django.contrib import admin
from .models import RecognitionPrediction


@admin.register(RecognitionPrediction)
class RecognitionPredictionAdmin(admin.ModelAdmin):
    list_display = ("predicted_sign", "model_name", "confidence", "latency_ms", "user", "created_at")
    list_filter = ("model_name", "created_at")
    search_fields = ("predicted_sign", "user__username")
