#!/usr/bin/env python3
"""Run paired CNN-LSTM and temporal-ViT experiments and write honest artifacts."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REQUIRED_MODULES = {"torch": "torch", "torchvision": "torchvision", "cv2": "opencv-python-headless", "numpy": "numpy", "pandas": "pandas", "sklearn": "scikit-learn", "psutil": "psutil"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="experiment_config.json")
    p.add_argument("--manifest")
    p.add_argument("--dataset-root")
    p.add_argument("--results-dir")
    p.add_argument("--models", nargs="+", choices=("cnn_lstm", "vit_temporal"), default=["cnn_lstm", "vit_temporal"])
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--num-frames", type=int)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    return p.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for arg, key in ((args.manifest, "manifest_file"), (args.dataset_root, "dataset_root"), (args.results_dir, "results_dir"), (args.epochs, "epochs"), (args.batch_size, "batch_size"), (args.num_frames, "num_frames"), (args.learning_rate, "learning_rate"), (args.device, "device")):
        if arg is not None: config[key] = arg
    seeds = args.seeds or config.get("seeds") or [config.get("seed", 42)]
    config["seeds"] = seeds
    results_dir = (ROOT / config.get("results_dir", "results")).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "run_log.txt"

    from training.environment import hardware_info, software_info
    hardware, software = hardware_info(), software_info()
    config["_hardware"], config["_software"] = hardware, software
    snapshot = results_dir / "experiment_config.json"
    snapshot.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    blockers = []
    missing = [f"{package} (Python module {module})" for module, package in REQUIRED_MODULES.items() if importlib.util.find_spec(module) is None]
    if missing:
        blockers.append("Missing ML dependencies: " + ", ".join(missing) + ". Install with: python -m pip install -r requirements-ml.txt")
    manifest = (ROOT / config["manifest_file"]).resolve()
    annotation = (ROOT / config["annotation_file"]).resolve()
    dataset_root = (ROOT / config["dataset_root"]).resolve()
    if not manifest.is_file():
        if annotation.is_file():
            blockers.append(f"Manifest missing: {manifest}. Build it after placing the official WLASL clips under {dataset_root}/videos/<video_id>.mp4 using: python datasets/build_manifest.py --annotations {annotation.relative_to(ROOT)} --dataset-root {dataset_root.relative_to(ROOT)} --classes {config['num_classes']} --seed {config.get('random_seed', 42)} --split-mode {config.get('split_mode', 'provided')}")
        else:
            blockers.append(f"WLASL annotations and manifest are missing. Place WLASL_v0.3.json at {annotation} and authorized WLASL video clips under {dataset_root}/videos/<video_id>.mp4; review the official WLASL C-UDA terms and request video access from the maintainers first.")
    elif not dataset_root.is_dir():
        blockers.append(f"Dataset root is missing: {dataset_root}")

    log_lines = [f"\n[{datetime.now(timezone.utc).isoformat()}] Experiment invocation", f"Python: {sys.version.split()[0]}", f"Hardware: {json.dumps(hardware, sort_keys=True)}", f"Software: {json.dumps(software, sort_keys=True)}", f"Models: {', '.join(args.models)}; seeds: {seeds}"]
    statuses = []
    if blockers:
        log_lines.extend(["Training not started; preflight requirements are incomplete:", *[f"- {item}" for item in blockers]])
        for model in args.models:
            for seed in seeds: statuses.append({"model": model, "seed": seed, "status": "not_started", "reason": blockers})
    else:
        try:
            from training.engine import load_manifest_context, run_experiment, validate_training_manifest
            frame, label_map = load_manifest_context(manifest)
            validate_training_manifest(frame, label_map, int(config["num_classes"]))
            missing_videos = [str(row.video_path) for row in frame.itertuples() if not (dataset_root / str(row.video_path)).is_file()]
            if missing_videos:
                blockers.append(f"{len(missing_videos)} manifest video paths do not exist under {dataset_root}; example: {missing_videos[0]}")
                raise RuntimeError(blockers[-1])
            for model in args.models:
                for seed in seeds:
                    run_id = f"{model}_seed_{seed}"
                    try:
                        # Run the two architectures sequentially with identical manifest and seed.
                        result = run_experiment(model, seed, config, ROOT, label_map)
                        out = results_dir / "metrics" / f"{run_id}_metrics.json"
                        out.parent.mkdir(parents=True, exist_ok=True)
                        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                        log_lines.append(f"COMPLETE {run_id}: test videos={result['dataset']['test_videos']} classes={result['dataset']['num_classes']} device={result['resources']['device']}")
                        statuses.append({"model": model, "seed": seed, "status": "complete", "metrics_file": str(out.relative_to(results_dir))})
                    except Exception as exc:
                        log_lines.append(f"FAILED {run_id}: {type(exc).__name__}: {exc}")
                        log_lines.append(traceback.format_exc())
                        statuses.append({"model": model, "seed": seed, "status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
                        # MPS runtime/kernel/OOM failures can be retried safely from scratch on CPU.
                        if config.get("device", "auto") in {"auto", "mps"} and hardware.get("accelerator") == "mps" and any(term in str(exc).lower() for term in ("mps", "out of memory", "not implemented")):
                            try:
                                log_lines.append(f"Retrying {run_id} from a fresh initialization on CPU after MPS failure.")
                                result = run_experiment(model, seed, config, ROOT, label_map, device_name="cpu")
                                out = results_dir / "metrics" / f"{run_id}_metrics.json"
                                out.parent.mkdir(parents=True, exist_ok=True)
                                result["resources"]["mps_fallback_reason"] = f"{type(exc).__name__}: {exc}"
                                out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                                statuses[-1] = {"model": model, "seed": seed, "status": "complete", "metrics_file": str(out.relative_to(results_dir)), "device_fallback": "cpu"}
                                log_lines.append(f"COMPLETE {run_id} on CPU fallback")
                            except Exception as fallback_exc:
                                log_lines.append(f"CPU fallback FAILED {run_id}: {type(fallback_exc).__name__}: {fallback_exc}")
                                log_lines.append(traceback.format_exc())
        except Exception as exc:
            blockers.append(f"Experiment preflight failed: {type(exc).__name__}: {exc}")
            log_lines.append(blockers[-1])
            log_lines.append(traceback.format_exc())
            if not statuses:
                statuses = [{"model": model, "seed": seed, "status": "not_started", "reason": blockers} for model in args.models for seed in seeds]

    status_doc = {"generated_at": datetime.now(timezone.utc).isoformat(), "status": "complete" if statuses and all(x["status"] == "complete" for x in statuses) else "incomplete", "blockers": blockers, "runs": statuses}
    (results_dir / "experiment_status.json").write_text(json.dumps(status_doc, indent=2) + "\n", encoding="utf-8")
    with log_path.open("a", encoding="utf-8") as handle: handle.write("\n".join(log_lines) + "\n")
    from research.reporting import generate_outputs
    docx = generate_outputs(results_dir, config)
    if docx:
        with log_path.open("a", encoding="utf-8") as handle: handle.write(f"Editable DOCX report generated: {docx.relative_to(ROOT)}\n")
    else:
        with log_path.open("a", encoding="utf-8") as handle: handle.write("Editable DOCX report skipped: python-docx is not installed. Install with: python -m pip install -r requirements-ml.txt\n")
    print(f"Experiment status: {status_doc['status']}; completed runs: {sum(x['status'] == 'complete' for x in statuses)}. See {log_path.relative_to(ROOT)} and {results_dir.relative_to(ROOT)}/experimental_results.md")
    return 0 if status_doc["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
