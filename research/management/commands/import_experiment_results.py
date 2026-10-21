import json
from pathlib import Path

from django.core.management.base import BaseCommand
from research.models import ExperimentRun


class Command(BaseCommand):
    help = "Import completed metrics and incomplete run statuses from results/ into Django."

    def add_arguments(self, parser):
        parser.add_argument("--results-dir", default="results")

    def handle(self, *args, **options):
        root = Path(options["results_dir"])
        imported = 0
        for path in sorted((root / "metrics").glob("*.json")) if (root / "metrics").is_dir() else []:
            result = json.loads(path.read_text(encoding="utf-8"))
            if result.get("status") != "complete":
                continue
            run_id = f"{result['model']}_seed_{result['seed']}"
            ExperimentRun.objects.update_or_create(run_id=run_id, defaults={
                "model_name": result["model"], "seed": int(result["seed"]), "status": "complete",
                "dataset_name": result.get("dataset", {}).get("name", ""), "config": {"config": result.get("config", {}), "training": result.get("training", {}), "dataset": result.get("dataset", {})},
                "metrics": result.get("metrics", {}), "per_class": result.get("per_class", []), "resources": result.get("resources", {}), "error": "",
            })
            imported += 1
        status_path = root / "experiment_status.json"
        if status_path.is_file():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            for item in status.get("runs", []):
                if item.get("status") == "complete":
                    continue
                run_id = f"{item['model']}_seed_{item['seed']}"
                ExperimentRun.objects.update_or_create(run_id=run_id, defaults={
                    "model_name": item["model"], "seed": int(item["seed"]), "status": item["status"],
                    "error": "; ".join(item.get("reason", [])) if isinstance(item.get("reason"), list) else str(item.get("reason", "")),
                })
        self.stdout.write(self.style.SUCCESS(f"Imported or updated {imported} completed experiment record(s)."))
