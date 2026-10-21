from django.contrib import admin
from .models import ExperimentRun


@admin.register(ExperimentRun)
class ExperimentRunAdmin(admin.ModelAdmin):
    list_display = ("run_id", "model_name", "seed", "status", "dataset_name", "imported_at")
    list_filter = ("model_name", "status", "dataset_name")
    search_fields = ("run_id", "error")
    readonly_fields = ("imported_at",)
