# SignBridge — Experimental Results

**Study:** A Comparative Study of CNN-LSTM and Vision Transformer Models for Real-Time Sign Language Recognition
**Run dates:** 20–21 October 2026 · **Hardware:** NVIDIA GeForce RTX 3060 Laptop GPU (6 GB), AMD Ryzen 16-thread CPU, Windows 11 · **Software:** Python 3.12.3, PyTorch 2.11.0 + CUDA 12.8, torchvision 0.26.0

Two experiments were run with the **same models, code and training recipe**; only the training data differs.

| | Experiment 1 | Experiment 2 (current app models) |
|---|---|---|
| Training data | WLASL-100 only (691 clips) | WLASL-100 + ASL Citizen (2,732 clips, 93 signers) |
| Main test set | WLASL-100 test (158 clips; signers overlap with training) | **ASL Citizen test (1,705 clips, 11 signers never seen in training)** |
| Also tested on | ASL Citizen sample via the web app | WLASL-100 test (same 158 clips as Exp. 1) |
| Raw files | [`results_wlasl100/`](results_wlasl100/) | [`results/`](results/) |
| Config | [`experiment_config_wlasl100.json`](experiment_config_wlasl100.json) | [`experiment_config.json`](experiment_config.json) |

All numbers are **mean ± sample standard deviation over 3 seeds (42, 123, 2026)**, measured on held-out test clips with the checkpoint chosen on the validation split. Random guessing is 1 % top-1 / 5 % top-5.

## 1. Headline results

### Experiment 2 — new signers (ASL Citizen test, 1,705 clips, 98 words, 11 unseen signers)

| Metric | CNN-LSTM | Vision Transformer | Better |
|---|---:|---:|---|
| Top-1 accuracy (%) | 16.60 ± 1.18 | **31.40 ± 1.30** | ViT |
| Top-5 accuracy (%) | 41.88 ± 1.58 | **60.49 ± 2.04** | ViT |
| Macro precision (%) | 15.98 ± 2.14 | **32.86 ± 2.59** | ViT |
| Macro recall (%) | 13.37 ± 1.62 | **29.33 ± 1.34** | ViT |
| Macro F1-score (%) | 12.87 ± 1.76 | **28.84 ± 1.68** | ViT |
| Inference latency (ms / clip, model only) | **31.8 ± 0.2** | 143.2 ± 3.9 | CNN-LSTM |
| Throughput (sampled frames / s) | **503.1 ± 2.5** | 111.8 ± 3.0 | CNN-LSTM |
| Checkpoint size (MB) | **101.1** | 381.7 | CNN-LSTM |
| Total / trainable parameters | **26.4 M / 2.9 M** | 100.1 M / 14.3 M | CNN-LSTM |
| Head training time (s, cached features) | **52.8 ± 4.6** | 99.6 ± 5.7 | CNN-LSTM |

### Effect of adding ASL Citizen (same 158 WLASL-100 test clips in both experiments)

| Metric | CNN-LSTM Exp. 1 → Exp. 2 | ViT Exp. 1 → Exp. 2 |
|---|---:|---:|
| Top-1 accuracy (%) | 11.39 → **21.10 ± 2.92** | 22.36 → **34.39 ± 1.46** |
| Top-5 accuracy (%) | 27.43 → **47.05 ± 2.99** | 49.37 → **61.81 ± 2.56** |
| Macro F1-score (%) | 8.56 → **18.01 ± 2.47** | 18.64 → **31.14 ± 1.61** |

### Through the real web app (upload form at `/recognize/`, seed-42 models)

Each clip was posted to the running Django server exactly as the "Choose video clip" button does ([`experiments/test_upload_endpoint.py`](experiments/test_upload_endpoint.py)).

| Test clips | Model | Exp. 1 top-1 / top-3 | Exp. 2 top-1 / top-3 |
|---|---|---:|---:|
| WLASL-100 test (158) | ViT | 21.5 % / 38.6 % | **33.5 % / 51.9 %** |
| WLASL-100 test (158) | CNN-LSTM | 13.3 % / 22.2 % | **22.2 % / 39.9 %** |
| ASL Citizen sample, 4 per word, new signers (392) | ViT | 6.1 % / 13.3 % | **23.0 % / 45.4 %** |
| ASL Citizen sample, 4 per word, new signers (392) | CNN-LSTM | 2.3 % / 6.9 % | **12.0 % / 25.8 %** |

Median request time (upload → answer, including video decoding) was 182 ms for the ViT and 103 ms for the CNN-LSTM. Exp. 1 web numbers were measured before a small preprocessing fix (the app now resizes frames with the same INTER_AREA method as training); the fix changed Exp. 2 web results by at most ±4 points. The web result for WLASL (33.5 %) is identical to the pipeline's seed-42 result, confirming the app processes video exactly like training. The 392-clip sample is harder than the full test set because it includes 4 clips of every word, including the hardest ones.

## 2. Per-seed results

**Experiment 2** (ASL Citizen test = main; WLASL test = extra)

| Model | Seed | Best epoch | Val top-1 | ASLC top-1 | ASLC top-5 | ASLC F1 | WLASL top-1 | WLASL top-5 | Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN-LSTM | 42 | 28 | 17.67 | 16.72 | 42.76 | 13.21 | 22.78 | 49.37 | 32.0 |
| CNN-LSTM | 123 | 20 | 17.67 | 15.37 | 40.06 | 10.97 | 17.72 | 43.67 | 31.7 |
| CNN-LSTM | 2026 | 26 | 19.29 | 17.71 | 42.82 | 14.43 | 22.78 | 48.10 | 31.7 |
| ViT | 42 | 27 | 31.66 | 30.21 | 58.24 | 27.38 | 33.54 | 58.86 | 139.5 |
| ViT | 123 | 32 | 34.76 | 31.20 | 61.00 | 28.47 | 33.54 | 63.29 | 143.0 |
| ViT | 2026 | 27 | 35.35 | 32.79 | 62.23 | 30.67 | 36.08 | 63.29 | 147.2 |

**Experiment 1** (WLASL-100 test)

| Model | Seed | Best epoch | Val top-1 | Test top-1 | Test top-5 | Macro F1 | Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN-LSTM | 42 | 30 | 13.92 | 11.39 | 29.75 | 9.32 | 30.5 |
| CNN-LSTM | 123 | 39 | 12.03 | 10.13 | 25.32 | 7.53 | 32.1 |
| CNN-LSTM | 2026 | 24 | 15.19 | 12.66 | 27.22 | 8.82 | 32.6 |
| ViT | 42 | 17 | 25.32 | 24.05 | 48.73 | 20.05 | 127.7 |
| ViT | 123 | 29 | 22.15 | 23.42 | 51.90 | 19.95 | 129.0 |
| ViT | 2026 | 14 | 21.52 | 19.62 | 47.47 | 15.92 | 128.3 |

## 3. Interpretation

- **Accuracy:** the temporal Vision Transformer beats the CNN-LSTM on every test set and every seed — about **2× the top-1 accuracy and macro F1**. On new signers the gap is ~15 points top-1 (31.4 % vs 16.6 %), more than 10× the seed-to-seed standard deviation.
- **Speed and size:** the CNN-LSTM is about **4.5× faster** (32 ms vs 143 ms per 16-frame clip) and **3.8× smaller**. Both run faster than real time on the test GPU.
- **Trade-off:** this supports the synopsis' exploratory hypothesis — CNN-LSTM gives lower latency, the temporally modelled ViT gives higher accuracy.
- **Generalisation to unseen signers (objective 7):** models trained on WLASL-100 alone almost completely failed on new signers recorded on webcams (6 % ViT, 2 % CNN-LSTM top-1 on the ASL Citizen sample) and collapsed onto a few favourite answers (e.g. "computer"). Adding 34 more signers from ASL Citizen raised new-signer top-1 to 31 % (ViT) and 17 % (CNN-LSTM). **Signer and recording diversity in the training data mattered more than any hyper-parameter change that was tried.**
- **Context:** the WLASL paper (Li et al., 2020) reports 25.97 % top-1 on WLASL-100 for its frame-CNN + RNN baseline (VGG-GRU) using all videos. Higher literature numbers (I3D ≈ 65.9 %, Pose-TGCN ≈ 55.4 %) come from 3D-CNN or pose-based models, outside this study's scope.
- **Statistics:** with 3 seeds a formal significance test has low power, so results are reported as mean ± standard deviation and no p-value is claimed. The ViT > CNN-LSTM accuracy difference is consistent across all 3 seeds and both test sets.

## 4. Protocol

| Item | Setting (same for both models and both experiments unless noted) |
|---|---|
| Vocabulary | The 100 WLASL-100 words (`datasets/WLASL-100/label_map.json`) |
| WLASL-100 data | 1,007 of 2,038 annotated clips available (Hugging Face mirror Voxel51/WLASL); stratified per-word split, seed 42: 691 train / 158 val / 158 test; signers overlap |
| ASL Citizen data (Exp. 2) | Official splits, words matched to our vocabulary (numbered variants like BED1/BED2 included; fingerspelled W.H.A.T excluded): 2,041 train / 521 val / 1,705 test; 98 words ("forget", "short" not present); test signers never in train/val |
| Input | 16 frames sampled evenly from each clip, resized to 224×224 (INTER_AREA), ImageNet normalisation |
| CNN-LSTM | ResNet-50 (ImageNet, frozen) → 2-layer LSTM (hidden 256) → linear |
| ViT | ViT-B/16 (ImageNet, frozen) → sinusoidal time encoding → 2-layer temporal Transformer (8 heads) → mean pool → linear |
| Training | AdamW lr 3e-4, weight decay 1e-4, dropout 0.3, label smoothing 0.1, batch 32, ≤150 epochs, ReduceLROnPlateau (on val top-1), early stopping patience 25 |
| Augmentation | 6 cached views per training clip (1 clean + 5 with time jitter, 80–100 % crop, horizontal flip); one random view per epoch |
| Checkpoint selection | Highest validation top-1 (ties → lower validation loss) |
| Macro metrics | Averaged over the words present in each test set |
| Latency / FPS | Model forward pass only, GPU-synchronised, warm-up excluded; decoding excluded (web request times above include it) |

## 5. Limitations

1. Only about half of WLASL-100 was obtainable; absolute WLASL numbers are not directly comparable with papers that use the full set.
2. The WLASL-100 split is not signer-independent. The ASL Citizen test set is, and is the main result of Experiment 2.
3. Backbones are frozen ImageNet models; end-to-end fine-tuning was tried in Experiment 1 and did not improve validation accuracy.
4. American Sign Language only; no Indian Sign Language and no fingerspelled letters.
5. Latency was measured on one GPU; CPU-only performance will be slower.
6. ASL Citizen is licensed by Microsoft Research for non-commercial research only; its videos are not included in this repository and must not be shown in presentations.

## 6. Files

| File | Contents |
|---|---|
| `results/model_comparison.csv` | Exp. 2 summary on ASL Citizen test |
| `results/model_comparison_test_wlasl.csv` | Exp. 2 summary on WLASL-100 test |
| `results/experimental_results.md` / `.docx` | Auto-generated Exp. 2 report with figures |
| `results/metrics/*.json` | Per-run metrics, per-word scores, confusion matrix, config, hardware |
| `results/curves/`, `results/figures/` | Training curves and charts |
| `results/run_log.txt` | Append-only log of every invocation |
| `results_wlasl100/` | The same files for Experiment 1 |
| `datasets/WLASL-100/manifest.csv` | WLASL-100 clip list and split |
