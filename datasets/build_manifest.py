#!/usr/bin/env python3
"""Build a reproducible WLASL-100 clip manifest from the official annotations."""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

try:
    from sklearn.model_selection import GroupShuffleSplit
except ImportError:
    GroupShuffleSplit = None


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def find_video(dataset_root, video_id):
    root = Path(dataset_root)
    for base in (root / "videos", root / "clips", root):
        for suffix in (".mp4", ".avi", ".mov", ".mkv"):
            path = base / f"{video_id}{suffix}"
            if path.is_file():
                # videos/ is normally the pre-trimmed clip output of the WLASL start kit.
                trimmed = base.name in {"videos", "clips"}
                return path, trimmed
    matches = sorted(path for path in root.rglob(f"{video_id}.*") if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"})
    if matches:
        path = matches[0]
        return path, path.parent.name in {"videos", "clips"}
    return None, False


def normalize_split(value):
    value = str(value or "").strip().lower()
    aliases = {"validation": "val", "valid": "val", "dev": "val", "training": "train", "testing": "test"}
    value = aliases.get(value, value)
    return value if value in {"train", "val", "test"} else ""


def group_split(rows, seed, test_size=0.15, val_size=0.15):
    if GroupShuffleSplit is None:
        raise RuntimeError("scikit-learn is required for signer-independent splitting. Install requirements-ml.txt.")
    signers = [row["signer_id"] for row in rows]
    if any(not signer for signer in signers):
        raise RuntimeError("Signer-independent splitting requires signer_id for every video; missing signer metadata was found.")
    all_indices = list(range(len(rows)))
    test_splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(test_splitter.split(all_indices, groups=signers))
    trainval_rows = [rows[i] for i in trainval_idx]
    trainval_signers = [rows[i]["signer_id"] for i in trainval_idx]
    relative_val_size = val_size / (1.0 - test_size)
    val_splitter = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=seed + 1)
    train_idx, val_idx = next(val_splitter.split(trainval_rows, groups=trainval_signers))
    for i in trainval_idx[train_idx]:
        rows[i]["split"] = "train"
    for i in trainval_idx[val_idx]:
        rows[i]["split"] = "val"
    for i in test_idx:
        rows[i]["split"] = "test"
    labels = {row["label"] for row in rows}
    missing = {split: sorted(labels - {r["label"] for r in rows if r["split"] == split}) for split in ("train", "val", "test")}
    missing = {split: values for split, values in missing.items() if values}
    if missing:
        details = "; ".join(f"{split}: {len(labels_)} missing classes" for split, labels_ in missing.items())
        raise RuntimeError(
            "The signer-independent split does not contain every class in every partition "
            f"({details}). Use the official provided split if it is signer-independent, or review the dataset split protocol."
        )


def stratified_split(rows, seed, test_size=0.15, val_size=0.15):
    """Per-class random split guaranteeing every class in train/val/test. Signers may overlap."""
    import random
    rng = random.Random(seed)
    by_label = {}
    for row in sorted(rows, key=lambda r: r["video_id"]):
        by_label.setdefault(row["label"], []).append(row)
    for label, group in by_label.items():
        if len(group) < 3:
            raise RuntimeError(f"Class {label!r} has {len(group)} clip(s); stratified splitting needs at least 3.")
        rng.shuffle(group)
        n_test = max(1, round(len(group) * test_size))
        n_val = max(1, round(len(group) * val_size))
        for i, row in enumerate(group):
            row["split"] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="datasets/WLASL-100/WLASL_v0.3.json")
    parser.add_argument("--dataset-root", default="datasets/WLASL-100")
    parser.add_argument("--output", default="datasets/WLASL-100/manifest.csv")
    parser.add_argument("--classes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-mode", choices=("provided", "signer-independent", "stratified"), default="provided")
    args = parser.parse_args()

    annotation_path = Path(args.annotations)
    if not annotation_path.is_file():
        raise SystemExit(
            f"Annotations not found: {annotation_path}\n"
            "Get WLASL_v0.3.json from the official WLASL repository after reviewing its C-UDA terms."
        )
    with annotation_path.open(encoding="utf-8") as handle:
        annotations = json.load(handle)
    class_counts = Counter({entry["gloss"]: len(entry.get("instances", [])) for entry in annotations})
    selected_labels = [label for label, _ in sorted(class_counts.items(), key=lambda item: (-item[1], item[0]))[:args.classes]]
    selected_set = set(selected_labels)
    rows, missing = [], []
    root = Path(args.dataset_root).resolve()
    for entry in annotations:
        label = entry.get("gloss", "")
        if label not in selected_set:
            continue
        for instance in entry.get("instances", []):
            video_id = str(instance.get("video_id", ""))
            video_path, trimmed = find_video(root, video_id)
            if video_path is None:
                missing.append(video_id)
                continue
            rows.append({
                "video_id": video_id,
                "video_path": video_path.relative_to(root).as_posix(),
                "label": label,
                "signer_id": str(instance.get("signer_id", "")),
                "split": normalize_split(instance.get("split")) if args.split_mode == "provided" else "",
                "frame_start": 0 if trimmed else max(0, int(instance.get("frame_start", 1) or 1) - 1),
                "frame_end": -1 if trimmed else int(instance.get("frame_end", -1) or -1),
                "trimmed_clip": str(trimmed).lower(),
            })
    if not rows:
        raise SystemExit(f"No videos found under {root}. Expected files such as {root}/videos/<video_id>.mp4.")

    if args.split_mode == "stratified":
        stratified_split(rows, args.seed)
    elif args.split_mode == "signer-independent" or {row["split"] for row in rows} != {"train", "val", "test"}:
        group_split(rows, args.seed)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ("video_id", "video_path", "label", "signer_id", "split", "frame_start", "frame_end", "trimmed_clip")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    mapping = {label: i for i, label in enumerate(selected_labels)}
    output.with_name("label_map.json").write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} clips across {len(selected_labels)} selected glosses to {output}")
    print(f"Missing local video files: {len(missing)}")
    for split in ("train", "val", "test"):
        subset = [row for row in rows if row["split"] == split]
        print(f"{split}: {len(subset)} videos, {len({row['label'] for row in subset})} classes")
    if missing:
        print("Some annotation instances have no local video and were omitted. This is recorded by the manifest size.")


if __name__ == "__main__":
    main()
