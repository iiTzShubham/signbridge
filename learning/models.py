from django.conf import settings
from django.db import models


class Lesson(models.Model):
    class Category(models.TextChoices):
        ALPHABET = "alphabet", "Alphabet"
        GREETINGS = "greetings", "Greetings & introductions"
        VOCABULARY = "vocabulary", "Everyday words"

    title = models.CharField(max_length=140)
    slug = models.SlugField(max_length=160, unique=True)
    category = models.CharField(max_length=20, choices=Category.choices)
    summary = models.CharField(max_length=240)
    lesson_text = models.TextField(help_text="Lesson notes shown to the learner.")
    image_path = models.CharField(
        max_length=255,
        blank=True,
        help_text="Optional path under static/, for example images/isl-alphabet-chart.jpg.",
    )
    video_embed_url = models.URLField(blank=True)
    source_url = models.URLField()
    source_name = models.CharField(max_length=120, default="ISLRTC")
    duration_minutes = models.PositiveSmallIntegerField(default=5)
    sort_order = models.PositiveSmallIntegerField(default=0)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("sort_order", "title")

    def __str__(self):
        return self.title


class LearningProgress(models.Model):
    class Status(models.TextChoices):
        STARTED = "started", "Started"
        COMPLETED = "completed", "Completed"

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="learning_progress")
    lesson = models.ForeignKey(Lesson, on_delete=models.CASCADE, related_name="learner_progress")
    status = models.CharField(max_length=12, choices=Status.choices, default=Status.STARTED)
    practice_count = models.PositiveIntegerField(default=0)
    first_viewed_at = models.DateTimeField(auto_now_add=True)
    last_viewed_at = models.DateTimeField(auto_now=True)
    completed_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        ordering = ("-last_viewed_at",)
        constraints = [
            models.UniqueConstraint(fields=("user", "lesson"), name="unique_user_lesson_progress"),
        ]

    def __str__(self):
        return f"{self.user} · {self.lesson} · {self.status}"
