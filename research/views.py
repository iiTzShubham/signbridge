import json

from django.conf import settings
from django.http import FileResponse, Http404
from django.shortcuts import render

from .models import ExperimentRun


def comparison(request):
    runs = list(ExperimentRun.objects.filter(status="complete"))
    latest = {}
    for run in runs:
        latest.setdefault(run.model_name, run)

    metric_definitions = [
        ("Top-1 accuracy", "top1_accuracy_pct", "metrics", "%", "How often the model's first guess is correct."),
        ("Top-5 accuracy", "top5_accuracy_pct", "metrics", "%", "How often the correct sign appears in its five best guesses."),
        ("Macro precision", "macro_precision_pct", "metrics", "%", "How often predictions are right, averaged fairly across signs."),
        ("Macro recall", "macro_recall_pct", "metrics", "%", "How many examples the model finds, averaged across signs."),
        ("Macro F1-score", "macro_f1_pct", "metrics", "%", "A balanced summary of precision and recall."),
        ("Inference latency", "average_inference_latency_ms", "metrics", "ms / clip", "Model processing time for one clip; video loading is excluded."),
        ("Throughput", "fps", "metrics", "frames / sec", "Sampled frames processed per second during inference."),
        ("Checkpoint size", "checkpoint_size_mb", "resources", "MB", "Size of the saved model checkpoint."),
        ("Training time", "duration_seconds", "training", "sec", "Training time before held-out test evaluation."),
    ]
    table = []
    for label, key, section, unit, description in metric_definitions:
        values = {}
        for model_key, model_run in (("cnn", latest.get("cnn_lstm")), ("vit", latest.get("vit_temporal"))):
            if not model_run:
                values[model_key] = None
                continue
            source = model_run.metrics if section == "metrics" else model_run.resources if section == "resources" else model_run.config.get("training", {})
            values[model_key] = source.get(key)
        table.append({"label": label, "unit": unit, "description": description, **values})

    result_dir = settings.BASE_DIR / "results"
    status_file = result_dir / "experiment_status.json"
    try:
        status = json.loads(status_file.read_text(encoding="utf-8")) if status_file.is_file() else {}
    except (OSError, json.JSONDecodeError):
        status = {}
    blockers = status.get("blockers", [])
    state = "complete" if status.get("status") == "complete" else "needs_setup" if blockers or not status else "ready"
    blocker_text = " ".join(str(item) for item in blockers).lower()
    return render(request, "research/comparison.html", {
        "latest": latest,
        "table": table,
        "blockers": blockers,
        "complete_count": sum(item.get("status") == "complete" for item in status.get("runs", [])),
        "state": state,
        "needs_ml": "missing ml dependencies" in blocker_text,
        "needs_dataset": "wlasl" in blocker_text or "manifest" in blocker_text,
        "report_available": (result_dir / "experimental_results.md").is_file(),
        "docx_available": (result_dir / "experimental_results.docx").is_file(),
    })


def download(request, artifact):
    files = {
        "report": ("experimental_results.md", "text/markdown"),
        "comparison": ("model_comparison.csv", "text/csv"),
        "docx": ("experimental_results.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    }
    if artifact not in files:
        raise Http404
    filename, content_type = files[artifact]
    path = settings.BASE_DIR / "results" / filename
    if not path.is_file():
        raise Http404
    return FileResponse(path.open("rb"), as_attachment=True, filename=filename, content_type=content_type)
