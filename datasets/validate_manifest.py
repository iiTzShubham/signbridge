#!/usr/bin/env python3
"""Check a manifest before training: required columns, video files exist, every split has every word,
and how many signers appear in more than one split (signer overlap).
"""
import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Validate a prepared video manifest before model training.")
    parser.add_argument("--manifest", default="datasets/WLASL-100/manifest.csv")
    parser.add_argument("--dataset-root", default="datasets/WLASL-100")
    parser.add_argument("--classes", type=int, default=100)
    parser.add_argument("--require-signer-independent", action="store_true")
    args = parser.parse_args()
    manifest = Path(args.manifest)
    if not manifest.is_file():
        raise SystemExit(f"Manifest missing: {manifest}. Run datasets/build_manifest.py first.")
    frame = pd.read_csv(manifest).fillna("")
    required = {"video_id", "video_path", "label", "signer_id", "split"}
    if missing := required - set(frame.columns):
        raise SystemExit(f"Manifest is missing columns: {', '.join(sorted(missing))}")
    if frame.empty:
        raise SystemExit("Manifest contains no videos.")
    root = Path(args.dataset_root)
    absent = [str(root / path) for path in frame["video_path"] if not (root / path).is_file()]
    if absent:
        raise SystemExit(f"{len(absent)} manifest video file(s) are missing. First missing path: {absent[0]}")
    labels = set(frame["label"])
    if len(labels) != args.classes:
        raise SystemExit(f"Expected {args.classes} classes, found {len(labels)}.")
    if set(frame["split"]) != {"train", "val", "test"}:
        raise SystemExit("Manifest must have train, val, and test splits.")
    for split, group in frame.groupby("split"):
        missing_labels = labels - set(group["label"])
        print(f"{split}: {len(group)} videos, {group['label'].nunique()} classes")
        if missing_labels:
            raise SystemExit(f"Split {split!r} is missing {len(missing_labels)} class(es): {', '.join(sorted(missing_labels)[:10])}")
    overlap = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        left_signers = set(frame.loc[frame.split == left, "signer_id"]) - {""}
        right_signers = set(frame.loc[frame.split == right, "signer_id"]) - {""}
        overlap[(left, right)] = left_signers & right_signers
    for pair, shared in overlap.items():
        print(f"Signer overlap {pair[0]}/{pair[1]}: {len(shared)}")
    if args.require_signer_independent and any(overlap.values()):
        raise SystemExit("Signer overlap detected. Rebuild with --split-mode signer-independent or document the supplied split protocol.")
    print(f"Manifest valid: {len(frame)} videos, {len(labels)} classes")


if __name__ == "__main__":
    main()
