"""Generate research artifacts only from completed experiment result files."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

MODELS = ("cnn_lstm", "vit_temporal")
MODEL_LABELS = {"cnn_lstm": "CNN-LSTM", "vit_temporal": "Vision Transformer"}
METRICS = [
    ("Top-1 accuracy (%)", "top1_accuracy_pct", True, "metrics"),
    ("Top-5 accuracy (%)", "top5_accuracy_pct", True, "metrics"),
    ("Macro precision (%)", "macro_precision_pct", True, "metrics"),
    ("Macro recall (%)", "macro_recall_pct", True, "metrics"),
    ("Macro F1-score (%)", "macro_f1_pct", True, "metrics"),
    ("Average inference latency (ms/clip)", "average_inference_latency_ms", False, "metrics"),
    ("FPS (sampled frames/s)", "fps", True, "metrics"),
    ("Model checkpoint size (MB)", "checkpoint_size_mb", False, "resources"),
    ("Training duration (s)", "duration_seconds", False, "training"),
]


def _protocol_config(config):
    return {key: value for key, value in config.items() if key not in {"seed", "seeds"} and not key.startswith("_")}


def load_completed(results_dir: Path, config=None):
    records = []
    current_manifest_hash = None
    if config and config.get("manifest_file"):
        manifest = results_dir.parent / config["manifest_file"]
        if manifest.is_file():
            current_manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    for path in sorted((results_dir / "metrics").glob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if row.get("status") == "complete" and row.get("model") in MODELS:
            if config is not None and _protocol_config(row.get("config", {})) != _protocol_config(config):
                continue
            if current_manifest_hash and row.get("dataset", {}).get("manifest_sha256") != current_manifest_hash:
                continue
            row["_source"] = str(path.relative_to(results_dir))
            records.append(row)
    return records


def metric_value(record, key, section):
    return record.get(section, {}).get(key)


def summarize(records):
    summary = {}
    for model in MODELS:
        model_rows = [r for r in records if r["model"] == model]
        summary[model] = {"runs": len(model_rows), "values": {}}
        for label, key, _higher, section in METRICS:
            vals = [float(v) for r in model_rows if (v := metric_value(r, key, section)) is not None and math.isfinite(float(v))]
            summary[model]["values"][key] = {
                "mean": statistics.mean(vals) if vals else None,
                "std": statistics.stdev(vals) if len(vals) > 1 else None,
                "n": len(vals),
            }
    return summary


def generate_csv(results_dir: Path, summary):
    output = results_dir / "model_comparison.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Evaluation metric", "CNN-LSTM", "Vision Transformer", "Better result"])
        for label, key, higher, _section in METRICS:
            left = summary["cnn_lstm"]["values"][key]["mean"]
            right = summary["vit_temporal"]["values"][key]["mean"]
            if left is None or right is None:
                better = "Not measured"
            elif math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9):
                better = "Tie"
            else:
                better = MODEL_LABELS[("cnn_lstm" if left > right else "vit_temporal") if higher else ("cnn_lstm" if left < right else "vit_temporal")]
            writer.writerow([label, _format_value(summary["cnn_lstm"]["values"][key]), _format_value(summary["vit_temporal"]["values"][key]), better])
    return output


def _format_value(item):
    if item["mean"] is None:
        return "Not measured"
    suffix = f" ± {item['std']:.3f}" if item["std"] is not None else ""
    return f"{item['mean']:.3f}{suffix} (n={item['n']})"


def generate_markdown(results_dir: Path, config, records, summary):
    output = results_dir / "experimental_results.md"
    lines = [f"# {config.get('experiment_title', 'Experimental Results')}", "", "## 1. Experimental Setup", "",
             f"- Dataset: {config.get('dataset_name', 'Not configured')}",
             f"- Classes configured: {config.get('num_classes', 'Not configured')}",
             f"- Frames per clip: {config.get('num_frames', 'Not configured')}; image size: {config.get('image_size', 'Not configured')}",
             f"- Configured split: {config.get('split_mode', 'Not configured')}; seed(s): {config.get('seeds', [config.get('seed', 42)])}",
             f"- Training settings: epochs={config.get('epochs')}, batch size={config.get('batch_size')}, learning rate={config.get('learning_rate')}, pretrained={config.get('pretrained')}",
             f"- Environment: {json.dumps(config.get('_hardware', {}), ensure_ascii=False)}; software: {json.dumps(config.get('_software', {}), ensure_ascii=False)}", "",
             "The saved experiment configuration is `experiment_config.json`. Each complete JSON record contains the actual split counts, test-video count, signer overlap, device, training duration, and model resource measurements.", "",
             "## 2. Results", "", "| Evaluation metric | CNN-LSTM | Vision Transformer | Better result |", "|---|---:|---:|---|"]
    csv_path = results_dir / "model_comparison.csv"
    with csv_path.open(encoding="utf-8") as f:
        next(f, None)
        for row in csv.reader(f):
            if len(row) == 4:
                lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |")
    lines += ["", f"Completed runs: CNN-LSTM {summary['cnn_lstm']['runs']}; Vision Transformer {summary['vit_temporal']['runs']}.", "",
              "Metrics are means across successful seeds when there is more than one run; the accompanying `±` is sample standard deviation. A single run is preliminary and does not estimate run-to-run variability.", "",
              "## 3. Comparative Analysis", ""]
    if summary["cnn_lstm"]["runs"] and summary["vit_temporal"]["runs"]:
        lines.append("The table reports measured values for both architectures. Per-metric direction is used for the ‘Better result’ column; these measurements do not establish a universal deployment winner. Latency and FPS here time the model forward pass on the recorded device and exclude video decoding and preprocessing.")
    else:
        lines.append("A comparative conclusion is unavailable because both architectures do not have completed test evaluations. No model is designated as better without paired measured results.")
    lines += ["", "## 4. Discussion", "", "Differences may be associated with spatial feature extraction, temporal aggregation, model capacity, pretrained weights, and the dataset's signer and class distribution. These are possible explanations, not findings, unless supported by completed runs and the recorded protocol. The result files and training curves are the evidence for this implementation.", "",
              "## 5. Hypothesis Evaluation", ""]
    if len(records) >= 2 and summary["cnn_lstm"]["runs"] > 1 and summary["vit_temporal"]["runs"] > 1:
        lines.append("Multiple runs are available for both models. Mean and standard deviation are reported. No p-value is produced by this pipeline; a statistical test should only be added after checking paired seeds, distributional assumptions, and the selected unit of analysis.")
    elif summary["cnn_lstm"]["runs"] and summary["vit_temporal"]["runs"]:
        lines.append("Both models have measured runs, but only one or too few runs are available for variability analysis. Any observed difference is preliminary; statistical significance is not claimed.")
    else:
        lines.append("Inconclusive: one or both experiments have not produced a completed test evaluation. There is no evidence yet to accept or reject a performance-difference hypothesis.")
    lines += ["", "## 6. Limitations", "", "- Dataset availability, split integrity, class balance, and signer overlap must be verified from the recorded manifest and run metadata.", "- Results are limited to the configured WLASL-100 data, preprocessing, architecture settings, seeds, and available hardware.", "- The latency measure excludes decoding and preprocessing and should not be interpreted as end-to-end real-time performance.", "- Statistical significance is not reported unless an appropriate analysis is separately implemented and justified.", ""]
    status_path = results_dir / "experiment_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
    if status.get("blockers") or any(run.get("status") != "complete" for run in status.get("runs", [])):
        lines += ["### Incomplete experiment blockers", ""]
        lines.extend(f"- {item}" for item in status.get("blockers", []))
        for run in status.get("runs", []):
            if run.get("status") != "complete":
                reason = run.get("reason", "No reason recorded")
                if isinstance(reason, list): reason = "; ".join(reason)
                lines.append(f"- {run.get('model')} seed {run.get('seed')}: {run.get('status')} — {reason}")
        lines.append("")
    lines += ["## 7. Conclusion", "", "No research conclusion can be drawn until both models complete evaluation on the shared held-out test split. This report intentionally leaves unavailable metrics as ‘Not measured’ and does not import published benchmark values into project results.", "",
              "## Published literature (separate from project measurements)", "", "No published benchmark numbers are included in this report. Add literature results only with their original citation, dataset, and evaluation conditions; they must remain distinct from these implementation measurements.", "",
              "## Figures", ""]
    figures = sorted((results_dir / "figures").glob("*.png")) if (results_dir / "figures").is_dir() else []
    if figures:
        lines.extend(f"![{p.stem.replace('_', ' ').title()}](figures/{p.name})\n\n*Figure: {p.stem.replace('_', ' ').title()} generated from completed experiment records.*\n" for p in figures)
    else:
        lines.append("No figures generated: there are no completed metric records from which to plot results.")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def generate_figures(results_dir: Path, records):
    if not records:
        return []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return []
    figures = results_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    summary = summarize(records)
    made = []
    def bar(name, title, key, section="metrics", ylabel=None):
        vals = [summary[m]["values"][key]["mean"] if section != "resources" else summary[m]["values"].get(key, {}).get("mean") for m in MODELS]
        if not any(v is not None for v in vals): return
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.bar([MODEL_LABELS[m] for m in MODELS], [v or 0 for v in vals], color=["#3977b8", "#ef9860"])
        ax.set_title(title); ax.set_ylabel(ylabel or title); ax.grid(axis="y", alpha=.25); fig.tight_layout()
        path = figures / f"{name}.png"; fig.savefig(path, dpi=220); plt.close(fig); made.append(path)
    for name, title, key, ylabel in (("accuracy_comparison", "Top-1 accuracy comparison", "top1_accuracy_pct", "Accuracy (%)"), ("f1_comparison", "Macro precision, recall, and F1", "macro_f1_pct", "Score (%)"), ("latency_comparison", "Model-only inference latency", "average_inference_latency_ms", "Latency (ms/clip)"), ("fps_comparison", "Model-only throughput", "fps", "Sampled frames per second")):
        bar(name, title, key, ylabel=ylabel)
    score_keys = (("Macro precision", "macro_precision_pct"), ("Macro recall", "macro_recall_pct"), ("Macro F1", "macro_f1_pct"))
    if any(summary[m]["values"][k]["mean"] is not None for m in MODELS for _, k in score_keys):
        fig, ax = plt.subplots(figsize=(7, 4.5)); x = np.arange(len(score_keys)); width = .34
        for offset, model in enumerate(MODELS):
            values = [summary[model]["values"][key]["mean"] or 0 for _, key in score_keys]
            ax.bar(x + (offset - .5) * width, values, width, label=MODEL_LABELS[model])
        ax.set(title="Macro precision, recall, and F1 comparison", ylabel="Score (%)", xticks=x, xticklabels=[name for name, _ in score_keys]); ax.legend(); ax.grid(axis="y", alpha=.25); fig.tight_layout()
        path = figures / "f1_comparison.png"; fig.savefig(path, dpi=220); plt.close(fig); made.append(path)
    for name, title, key in (("model_size_comparison", "Checkpoint size", "checkpoint_size_mb"),):
        bar(name, title, key, section="resources", ylabel="Size (MB)")
    # Curves are emitted only for recorded history CSVs belonging to complete records.
    histories = []
    for record in records:
        p = results_dir.parent / record.get("training_history", "")
        if p.is_file(): histories.append((record, p))
    if histories:
        import pandas as pd
        for name, column, title, ylabel in (("validation_performance", "val_top1_accuracy_pct", "Validation top-1 accuracy", "Accuracy (%)"),):
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for record, p in histories:
                frame = pd.read_csv(p)
                ax.plot(frame["epoch"], frame[column], label=f"{MODEL_LABELS[record['model']]} seed {record['seed']}")
            ax.set(title=title, xlabel="Epoch", ylabel=ylabel); ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
            path = figures / f"{name}.png"; fig.savefig(path, dpi=220); plt.close(fig); made.append(path)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for record, p in histories:
            frame = pd.read_csv(p)
            ax.plot(frame["epoch"], frame["train_loss"], label=f"{MODEL_LABELS[record['model']]} train seed {record['seed']}")
            ax.plot(frame["epoch"], frame["val_loss"], linestyle="--", label=f"{MODEL_LABELS[record['model']]} validation seed {record['seed']}")
        ax.set(title="Training and validation loss by epoch", xlabel="Epoch", ylabel="Cross-entropy loss"); ax.legend(fontsize=8); ax.grid(alpha=.25); fig.tight_layout()
        path = figures / "training_curves.png"; fig.savefig(path, dpi=220); plt.close(fig); made.append(path)
    for model in MODELS:
        candidates = [r for r in records if r["model"] == model]
        if not candidates: continue
        classes = candidates[0].get("class_names", [])
        matrices = [np.asarray(r.get("confusion_matrix"), dtype=float) for r in candidates if r.get("confusion_matrix")]
        if not matrices or any(m.shape != matrices[0].shape for m in matrices): continue
        normed = [m / np.maximum(m.sum(axis=1, keepdims=True), 1) for m in matrices]
        matrix = np.mean(normed, axis=0)
        fig, ax = plt.subplots(figsize=(8, 7)); im = ax.imshow(matrix, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
        ax.set(title=f"{MODEL_LABELS[model]} normalized confusion matrix (mean across {len(normed)} seed(s))", xlabel="Predicted class", ylabel="True class")
        if len(classes) <= 40: ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=6); ax.set_yticks(range(len(classes)), classes, fontsize=6)
        fig.colorbar(im, ax=ax, label="Row-normalized proportion"); fig.tight_layout()
        file_model = "vision_transformer" if model == "vit_temporal" else model
        path = figures / f"{file_model}_confusion_matrix.png"; fig.savefig(path, dpi=220); plt.close(fig); made.append(path)
    return made


def generate_docx(results_dir: Path, config, summary):
    try:
        from docx import Document
        from docx.shared import Inches
    except ImportError:
        return None
    doc = Document()
    doc.add_heading(config.get("experiment_title", "Experimental Results"), 0)
    md = (results_dir / "experimental_results.md").read_text(encoding="utf-8")
    section = None
    for line in md.splitlines():
        if line.startswith("## "):
            section = line[3:]; doc.add_heading(section, level=1)
        elif line.startswith("### "):
            doc.add_heading(line[4:], level=2)
        elif line.startswith("| Evaluation metric"):
            table = doc.add_table(rows=1, cols=4); table.style = "Light Shading Accent 1"
            for cell, value in zip(table.rows[0].cells, ("Evaluation metric", "CNN-LSTM", "Vision Transformer", "Better result")): cell.text = value
        elif line.startswith("|---"):
            continue
        elif line.startswith("| "):
            values = [v.strip() for v in line.strip("| ").split("|")]
            if len(values) == 4:
                cells = table.add_row().cells
                for cell, value in zip(cells, values): cell.text = value
        elif line.startswith("- "):
            doc.add_paragraph(line[2:], style="List Bullet")
        elif line and not line.startswith("!") and not line.startswith("*"):
            doc.add_paragraph(line)
    for image in sorted((results_dir / "figures").glob("*.png")) if (results_dir / "figures").is_dir() else []:
        doc.add_picture(str(image), width=Inches(6.2))
        p = doc.add_paragraph(f"Figure: {image.stem.replace('_', ' ').title()} generated from completed experiment records.")
        p.style = "Caption"
    path = results_dir / "experimental_results.docx"; doc.save(path); return path


def generate_outputs(results_dir: Path, config):
    results_dir.mkdir(parents=True, exist_ok=True)
    records = load_completed(results_dir, config)
    summary = summarize(records)
    for model, filename in (("cnn_lstm", "cnn_lstm_metrics.json"), ("vit_temporal", "vision_transformer_metrics.json")):
        runs = [r for r in records if r["model"] == model]
        payload = {"status": "complete" if runs else "not_measured", "model": model, "runs": len(runs), "metrics": summary[model]["values"], "run_files": [r["_source"] for r in runs]}
        (results_dir / filename).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    generate_figures(results_dir, records)
    generate_csv(results_dir, summary)
    generate_markdown(results_dir, config, records, summary)
    return generate_docx(results_dir, config, summary)
