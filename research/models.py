from django.db import models


class ExperimentRun(models.Model):
    STATUS_CHOICES = (("complete", "Complete"), ("failed", "Failed"), ("not_started", "Not started"))
    run_id = models.CharField(max_length=160, unique=True)
    model_name = models.CharField(max_length=40)
    seed = models.IntegerField()
    status = models.CharField(max_length=16, choices=STATUS_CHOICES)
    dataset_name = models.CharField(max_length=120, blank=True)
    config = models.JSONField(default=dict, blank=True)
    metrics = models.JSONField(default=dict, blank=True)
    per_class = models.JSONField(default=list, blank=True)
    resources = models.JSONField(default=dict, blank=True)
    error = models.TextField(blank=True)
    imported_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("-imported_at", "model_name", "seed")

    def __str__(self):
        return f"{self.model_name} seed {self.seed} ({self.status})"
