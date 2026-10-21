import csv
import gc
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset

from datasets.video_dataset import SignVideoDataset
from ml_models.factory import build_model
from .memory_monitor import PeakMemoryMonitor
from .utils import peak_memory_gib, seed_everything, select_device, synchronize


def load_manifest_context(manifest_path):
    import pandas as pd
    frame = pd.read_csv(manifest_path).fillna("")
    labels = sorted(frame["label"].astype(str).unique())
    mapping_path = Path(manifest_path).with_name("label_map.json")
    if mapping_path.is_file():
        saved = json.loads(mapping_path.read_text(encoding="utf-8"))
        if set(saved) == set(labels):
            label_to_index = {label: int(index) for label, index in saved.items()}
        else:
            label_to_index = {label: i for i, label in enumerate(labels)}
    else:
        label_to_index = {label: i for i, label in enumerate(labels)}
    return frame, label_to_index


def validate_training_manifest(frame, label_to_index, expected_classes):
    if len(label_to_index) != expected_classes:
        raise RuntimeError(f"Expected {expected_classes} classes in the manifest; found {len(label_to_index)}.")
    if not {"train", "val", "test"}.issubset(set(frame["split"])):
        raise RuntimeError("The manifest needs non-empty train, val, and test splits.")
    classes = set(label_to_index)
    for split in ("train", "val", "test"):
        found = set(frame.loc[frame.split == split, "label"].astype(str))
        if found != classes:
            raise RuntimeError(f"Split {split!r} does not contain all {len(classes)} classes ({len(classes-found)} missing).")


def _loader(config, root, manifest, split, label_to_index, seed, shuffle=False):
    dataset = SignVideoDataset(
        manifest=manifest,
        dataset_root=root,
        split=split,
        label_to_index=label_to_index,
        num_frames=int(config["num_frames"]),
        image_size=int(config["image_size"]),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=False,
        generator=generator if shuffle else None,
        persistent_workers=int(config.get("num_workers", 0)) > 0,
    )


class _ViewsDataset(Dataset):
    def __init__(self, base, views, seed):
        self.base, self.views, self.seed = base, views, seed

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        row = self.base.rows.iloc[index]
        label = self.base.label_to_index[str(row["label"])]
        return self.base.augmented_views(index, self.views, self.seed), label


class _FeatureDataset(Dataset):
    """Cached [views, clips, time, features]; training samples one random view per clip per epoch."""
    def __init__(self, features, labels, random_view):
        self.features, self.labels, self.random_view = features, labels, random_view

    def __len__(self):
        return self.features.shape[1]

    def __getitem__(self, index):
        view = int(torch.randint(self.features.shape[0], (1,))) if self.random_view else 0
        return self.features[view, index].float(), self.labels[index], ""


def _cached_features(model, config, root, manifest, split, label_to_index, views, device, cache_path):
    if cache_path.is_file():
        cached = torch.load(cache_path, weights_only=True)
        print(f"Loaded cached {split} features: {cache_path.name}", flush=True)
        return cached["features"], cached["labels"]
    base = SignVideoDataset(manifest=manifest, dataset_root=root, split=split, label_to_index=label_to_index,
                            num_frames=int(config["num_frames"]), image_size=int(config["image_size"]))
    loader = DataLoader(_ViewsDataset(base, views, int(config.get("augmentation_seed", 0))), batch_size=1,
                        num_workers=int(config.get("num_workers", 0)))
    features, labels = [], []
    started = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for clip_views, label in loader:
            features.append(model.encode_frames(clip_views[0].to(device)).half().cpu())
            labels.append(int(label))
    tensor = torch.stack(features, dim=1)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": tensor, "labels": label_tensor}, cache_path)
    print(f"Extracted {split} features {tuple(tensor.shape)} in {time.perf_counter() - started:.0f}s", flush=True)
    return tensor, label_tensor


def _epoch(model, loader, criterion, device, optimizer=None, forward=None):
    forward = forward or model
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    for videos, labels, _ in loader:
        videos = videos.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = forward(videos)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        count += labels.size(0)
    if count == 0:
        raise RuntimeError("Encountered an empty data split.")
    return total_loss / count, correct / count


def _training_loop(model_name, model, train_loader, val_loader, device, config, checkpoint_path, history_path, forward=None):
    criterion = nn.CrossEntropyLoss(label_smoothing=float(config.get("label_smoothing", 0.0)))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    select_by_accuracy = config.get("checkpoint_metric", "val_loss") == "val_top1"
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max" if select_by_accuracy else "min", patience=2, factor=0.5)
    best_val_loss = math.inf
    best_score = (-math.inf, -math.inf)
    best_epoch = 0
    patience = int(config.get("early_stopping_patience", 7))
    wait = 0
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("epoch", "train_loss", "train_top1_accuracy_pct", "val_loss", "val_top1_accuracy_pct", "learning_rate"))
        writer.writeheader()
        for epoch in range(1, int(config["epochs"]) + 1):
            started = time.perf_counter()
            train_loss, train_acc = _epoch(model, train_loader, criterion, device, optimizer, forward)
            val_loss, val_acc = _epoch(model, val_loader, criterion, device, forward=forward)
            scheduler.step(val_acc if select_by_accuracy else val_loss)
            writer.writerow({
                "epoch": epoch,
                "train_loss": f"{train_loss:.8f}",
                "train_top1_accuracy_pct": f"{train_acc * 100:.6f}",
                "val_loss": f"{val_loss:.8f}",
                "val_top1_accuracy_pct": f"{val_acc * 100:.6f}",
                "learning_rate": f"{optimizer.param_groups[0]['lr']:.10g}",
            })
            handle.flush()
            print(f"{model_name} epoch {epoch}/{config['epochs']} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_top1={val_acc*100:.2f}% elapsed={time.perf_counter()-started:.1f}s", flush=True)
            # The configured validation metric selects the checkpoint for both architectures (ties: lower val loss).
            score = (val_acc, -val_loss) if select_by_accuracy else (-val_loss, val_acc)
            if score > best_score:
                best_score = score
                best_val_loss = val_loss
                best_epoch = epoch
                wait = 0
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": model.state_dict(), "best_epoch": epoch, "best_val_loss": val_loss, "best_val_top1_pct": val_acc * 100}, checkpoint_path)
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping after {epoch} epochs; best validation checkpoint ({config.get('checkpoint_metric', 'val_loss')}) is epoch {best_epoch}.", flush=True)
                    break
    if not checkpoint_path.is_file():
        raise RuntimeError("Training ended without a validation-selected checkpoint.")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    return {"best_epoch": best_epoch, "best_val_loss": best_val_loss, "best_val_top1_pct": round(checkpoint.get("best_val_top1_pct", float("nan")), 4),
            "checkpoint_metric": config.get("checkpoint_metric", "val_loss"), "epochs_completed": epoch}


def _evaluate(model, loader, device, label_names, warmup_batches):
    model.eval()
    iterator = iter(loader)
    with torch.no_grad():
        for _ in range(warmup_batches):
            try:
                videos, _, _ = next(iterator)
            except StopIteration:
                break
            model(videos.to(device))
    synchronize(device)
    y_true, y_pred, y_prob, ids = [], [], [], []
    batch_times = []
    model_frame_count = 0
    with torch.no_grad():
        for videos, labels, video_ids in loader:
            videos = videos.to(device)
            synchronize(device)
            started = time.perf_counter()
            logits = model(videos)
            synchronize(device)
            elapsed = time.perf_counter() - started
            batch_times.extend([elapsed / len(labels)] * len(labels))
            model_frame_count += videos.shape[0] * videos.shape[1]
            probabilities = torch.softmax(logits, dim=1).cpu()
            y_true.extend(labels.tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
            y_prob.extend(probabilities.tolist())
            ids.extend(video_ids)
    if not y_true:
        raise RuntimeError("Test split is empty; no test metrics were measured.")
    top5 = min(5, len(label_names))
    top5_correct = sum(int(target in np.argsort(prob)[-top5:]) for target, prob in zip(y_true, y_prob)) / len(y_true)
    report = classification_report(
        y_true, y_pred, labels=list(range(len(label_names))), target_names=label_names,
        output_dict=True, zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    latency = float(np.mean(batch_times) * 1000)
    elapsed = float(sum(batch_times))
    frame_rate = float(model_frame_count / elapsed) if elapsed else None
    return {
        "test_video_ids": ids,
        "test_labels": y_true,
        "predicted_labels": y_pred,
        "top5_accuracy_pct": round(top5_correct * 100, 6) if len(label_names) >= 5 else None,
        "top1_accuracy_pct": round(report["accuracy"] * 100, 6),
        "macro_precision_pct": round(report["macro avg"]["precision"] * 100, 6),
        "macro_recall_pct": round(report["macro avg"]["recall"] * 100, 6),
        "macro_f1_pct": round(report["macro avg"]["f1-score"] * 100, 6),
        "per_class": [
            {"label": label, "precision_pct": round(report[label]["precision"] * 100, 6),
             "recall_pct": round(report[label]["recall"] * 100, 6),
             "f1_pct": round(report[label]["f1-score"] * 100, 6),
             "support": int(report[label]["support"])}
            for label in label_names
        ],
        "confusion_matrix": matrix.tolist(),
        "average_inference_latency_ms": round(latency, 6),
        "fps": round(frame_rate, 6) if frame_rate is not None else None,
        "inference_processed_frames": model_frame_count,
        "latency_method": "Mean per-clip model forward-pass time, batch latency divided by batch size; warm-up batches excluded; device synchronization used before and after timed batches. Video decoding and preprocessing are excluded. FPS = clips x sampled frames / summed model-only clip latency.",
    }


def run_experiment(model_name, seed, config, project_root, label_to_index, device_name=None):
    manifest = (project_root / config["manifest_file"]).resolve()
    dataset_root = (project_root / config["dataset_root"]).resolve()
    frame, _ = load_manifest_context(manifest)
    validate_training_manifest(frame, label_to_index, int(config["num_classes"]))
    seed_everything(seed)
    device = select_device(device_name or config.get("device", "auto"))
    print(f"{model_name} seed={seed} device={device}", flush=True)
    loaders = {
        split: _loader(config, dataset_root, manifest, split, label_to_index, seed, shuffle=(split == "train"))
        for split in ("train", "val", "test")
    }
    runtime_config = dict(config)
    runtime_config["seed"] = seed
    result_dir = project_root / config.get("results_dir", "results")
    checkpoint_path = result_dir / "checkpoints" / model_name / f"seed_{seed}.pth"
    history_path = result_dir / "curves" / model_name / f"seed_{seed}.csv"
    started = time.perf_counter()
    monitor = PeakMemoryMonitor(device).start()
    try:
        model = build_model(model_name, int(config["num_classes"]), runtime_config).to(device)
        train_loader, val_loader, forward, extraction_seconds = loaders["train"], loaders["val"], None, None
        if config.get("cache_features") and config.get("freeze_backbone", True):
            # The frozen backbone gives identical features every epoch, so compute them once and
            # train only the temporal head. Test evaluation below still runs the full model on raw video.
            tag = f"{int(config['num_frames'])}f_{int(config['image_size'])}px_{hashlib.sha256(manifest.read_bytes()).hexdigest()[:12]}"
            cache_dir = result_dir / "feature_cache"
            views = int(config.get("feature_views", 1))
            train_x, train_y = _cached_features(model, config, dataset_root, manifest, "train", label_to_index, views, device,
                                                cache_dir / f"{model_name}_train_{views}v_{tag}.pt")
            val_x, val_y = _cached_features(model, config, dataset_root, manifest, "val", label_to_index, 1, device,
                                            cache_dir / f"{model_name}_val_1v_{tag}.pt")
            extraction_seconds = time.perf_counter() - started
            started = time.perf_counter()  # training duration below covers head training only
            head_batch = int(config.get("head_batch_size", config["batch_size"]))
            generator = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(_FeatureDataset(train_x, train_y, True), batch_size=head_batch, shuffle=True, generator=generator)
            val_loader = DataLoader(_FeatureDataset(val_x, val_y, False), batch_size=head_batch)
            forward = model.forward_features
        training_stats = _training_loop(
            model_name, model, train_loader, val_loader, device,
            runtime_config, checkpoint_path, history_path, forward,
        )
        training_duration = time.perf_counter() - started
        evaluation = _evaluate(
            model, loaders["test"], device, list(label_to_index.keys()),
            int(config.get("warmup_inference_batches", 5)),
        )
        synchronize(device)
        monitor.stop()
        all_parameters = sum(parameter.numel() for parameter in model.parameters())
        trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        output = {
            "status": "complete",
            "model": model_name,
            "seed": seed,
            "dataset": {
                "name": config.get("dataset_name", "WLASL-100"),
                "manifest": str(manifest.relative_to(project_root)),
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "split_mode": config.get("split_mode", "provided"),
                "split_counts": {split: int((frame.split == split).sum()) for split in ("train", "val", "test")},
                "test_videos": len(evaluation["test_video_ids"]),
                "num_classes": len(label_to_index),
                "class_names": list(label_to_index.keys()),
                "signer_overlap": signer_overlap(frame),
            },
            "config": runtime_config,
            "training": {**training_stats, "duration_seconds": round(training_duration, 6),
                         "feature_cache": bool(forward), "feature_extraction_or_load_seconds": None if extraction_seconds is None else round(extraction_seconds, 3)},
            "metrics": {key: value for key, value in evaluation.items() if key not in {"per_class", "confusion_matrix", "test_video_ids", "test_labels", "predicted_labels"}},
            "per_class": evaluation["per_class"],
            "confusion_matrix": evaluation["confusion_matrix"],
            "class_names": list(label_to_index.keys()),
            "resources": {
                "parameter_count": all_parameters,
                "trainable_parameter_count": trainable_parameters,
                "checkpoint_size_mb": round(checkpoint_path.stat().st_size / (1024 ** 2), 6),
                "peak_memory_gib": peak_memory_gib(device, monitor.peak_rss_bytes, monitor.peak_accelerator_bytes),
                "peak_memory_method": "Peak sampled process RSS on CPU or framework allocator/driver allocation on the selected accelerator; sampled every 100 ms. This is not system-wide memory.",
                "device": str(device),
            },
            "hardware": config.get("_hardware", {}),
            "software": config.get("_software", {}),
            "checkpoint": str(checkpoint_path.relative_to(project_root)),
            "training_history": str(history_path.relative_to(project_root)),
        }
        return output
    finally:
        monitor.stop()
        try:
            del model
        except UnboundLocalError:
            pass
        if device.type == "mps":
            try:
                torch.mps.empty_cache()
            except (AttributeError, RuntimeError):
                pass
        gc.collect()


def signer_overlap(frame):
    result = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        left_ids = set(frame.loc[frame.split == left, "signer_id"].astype(str)) - {"", "nan", "None"}
        right_ids = set(frame.loc[frame.split == right, "signer_id"].astype(str)) - {"", "nan", "None"}
        result[f"{left}_vs_{right}"] = len(left_ids & right_ids)
    return result
