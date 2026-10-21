from django.conf import settings
from django.db import models


class Sign(models.Model):
    class Category(models.TextChoices):
        GREETING = "greeting", "Greetings"
        DAILY = "daily", "Everyday words"
        PEOPLE = "people", "People"
        LEARNING = "learning", "Learning"

    english_word = models.CharField(max_length=80, unique=True)
    hindi_word = models.CharField(max_length=80, blank=True)
    marathi_word = models.CharField(max_length=80, blank=True)
    description = models.CharField(max_length=240)
    category = models.CharField(max_length=20, choices=Category.choices, default=Category.DAILY)
    difficulty = models.CharField(max_length=24, default="Beginner")
    example_sentence = models.CharField(max_length=240, blank=True)
    image_path = models.CharField(max_length=255, blank=True)
    video_embed_url = models.URLField(blank=True)
    source_url = models.URLField(blank=True)
    sort_order = models.PositiveSmallIntegerField(default=0)

    class Meta:
        ordering = ("sort_order", "english_word")

    def __str__(self):
        return self.english_word


class TranslationHistory(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="translation_history", null=True, blank=True)
    input_text = models.TextField(max_length=1000)
    source_language = models.CharField(max_length=12)
    target_language = models.CharField(max_length=12)
    translated_text = models.TextField(max_length=1200)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("-created_at",)

    def __str__(self):
        return f"{self.source_language} → {self.target_language}: {self.input_text[:45]}"
