#!/usr/bin/env python3
"""Merge WLASL-100 and ASL Citizen into one clip list (manifest) for Experiment 2.

Why: WLASL-100 alone has ~7 training clips per word, mostly from a few websites, so the models
learned those backgrounds and signers instead of the signs. ASL Citizen adds 34 more signers
recorded on home webcams - much closer to what the live webcam page sees.

How the splits are assigned (each clip goes to exactly one split):
  train       = WLASL-100 train  + ASL Citizen train   (used to learn)
  val         = WLASL-100 val    + ASL Citizen val     (used to pick the best epoch)
  test        = ASL Citizen test  -> 11 signers who never appear in train/val (main result)
  test_wlasl  = WLASL-100 test    -> same test clips as Experiment 1, for comparison

The label map (word -> class number) is copied from WLASL-100 so class numbers stay the same.

Usage:
  python datasets/build_combined_manifest.py
"""
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
WLASL_MANIFEST = DATASETS / "WLASL-100" / "manifest.csv"
ASLC_CLIPS = DATASETS / "ASL-Citizen" / "clips.csv"   # written by experiments/download_asl_citizen_subset.py
OUTPUT = DATASETS / "combined" / "manifest.csv"
FIELDS = ("video_id", "video_path", "label", "signer_id", "split", "frame_start", "frame_end", "trimmed_clip", "source")


def main():
    for needed in (WLASL_MANIFEST, ASLC_CLIPS):
        if not needed.is_file():
            raise SystemExit(f"Missing {needed}. See docs/PROJECT_NOTES.md for how to create it.")

    rows = []
    # WLASL-100 clips: paths become relative to datasets/ (e.g. WLASL-100/videos/12345.mp4).
    with WLASL_MANIFEST.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append({
                **{key: row[key] for key in ("video_id", "label", "frame_start", "frame_end", "trimmed_clip")},
                "video_path": f"WLASL-100/{row['video_path']}",
                "signer_id": f"wlasl_{row['signer_id']}",
                "split": "test_wlasl" if row["split"] == "test" else row["split"],
                "source": "WLASL-100",
            })

    # ASL Citizen clips: already one sign per clip, so frame_start/end cover the whole video.
    with ASLC_CLIPS.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            video_path = f"ASL-Citizen/videos/{row['file']}"
            if not (DATASETS / video_path).is_file():
                continue  # skip anything that failed to download
            rows.append({
                "video_id": "aslc_" + Path(row["file"]).stem,
                "video_path": video_path,
                "label": row["label"],
                "signer_id": f"aslc_{row['participant']}",
                "split": row["split"],
                "frame_start": 0,
                "frame_end": -1,
                "trimmed_clip": "true",
                "source": "ASL-Citizen",
            })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    shutil.copyfile(WLASL_MANIFEST.with_name("label_map.json"), OUTPUT.with_name("label_map.json"))

    # Print a short summary so problems are easy to spot.
    print(f"Wrote {len(rows)} clips to {OUTPUT}")
    for split in ("train", "val", "test", "test_wlasl"):
        part = [r for r in rows if r["split"] == split]
        sources = Counter(r["source"] for r in part)
        print(f"  {split:<10} {len(part):>5} clips, {len({r['label'] for r in part}):>3} words, "
              f"{len({r['signer_id'] for r in part}):>3} signers  {dict(sources)}")
    signers = {split: {r["signer_id"] for r in rows if r["split"] == split} for split in ("train", "val", "test")}
    print(f"  ASL Citizen test signers also in train/val: {len(signers['test'] & (signers['train'] | signers['val']))} (should be 0)")
    labels = set(json.loads(OUTPUT.with_name("label_map.json").read_text(encoding="utf-8")))
    unknown = {r["label"] for r in rows} - labels
    if unknown:
        raise SystemExit(f"Labels not in the WLASL-100 label map: {sorted(unknown)}")


if __name__ == "__main__":
    main()
