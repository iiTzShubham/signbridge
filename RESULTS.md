# SignBridge — Experimental Results

**Study:** A Comparative Study of CNN-LSTM and Vision Transformer Models for Real-Time Sign Language Recognition
**Run date:** 21 October 2026 · **Hardware:** NVIDIA GeForce RTX 3060 Laptop GPU (6 GB), AMD Ryzen 16-thread CPU, Windows 11 · **Software:** Python 3.12.3, PyTorch 2.11.0 + CUDA 12.8, torchvision 0.26.0

All numbers below are measured on the held-out **test split (158 clips, 100 classes)** using the checkpoint selected on the validation split. Raw evidence is in [`results/`](results/).

## 1. Summary (mean ± sample std over 3 seeds: 42, 123, 2026)

| Metric | CNN-LSTM | Vision Transformer | Better |
|---|---:|---:|---|
| Top-1 accuracy (%) | 11.39 ± 1.27 | **22.36 ± 2.40** | ViT |
| Top-5 accuracy (%) | 27.43 ± 2.22 | **49.37 ± 2.28** | ViT |
| Macro precision (%) | 8.51 ± 1.36 | **19.61 ± 2.92** | ViT |
| Macro recall (%) | 10.33 ± 1.53 | **21.33 ± 2.57** | ViT |
| Macro F1-score (%) | 8.56 ± 0.92 | **18.64 ± 2.36** | ViT |
| Inference latency (ms / clip, model only) | **31.7 ± 1.1** | 128.3 ± 0.7 | CNN-LSTM |
| Throughput (sampled frames / s) | **504.6 ± 17.4** | 124.7 ± 0.6 | CNN-LSTM |
| Checkpoint size (MB) | **101.1** | 381.7 | CNN-LSTM |
| Total parameters | **26.4 M** | 100.1 M | CNN-LSTM |
| Trainable parameters (temporal head) | **2.9 M** | 14.3 M | CNN-LSTM |
| Head training time (s, cached features) | **16.0 ± 1.4** | 24.5 ± 3.9 | CNN-LSTM |

Random guessing is 1 % top-1 / 5 % top-5.

## 2. Per-seed results

| Model | Seed | Best epoch | Val top-1 (%) | Test top-1 (%) | Test top-5 (%) | Macro F1 (%) | Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN-LSTM | 42 | 30 | 13.92 | 11.39 | 29.75 | 9.32 | 30.5 |
| CNN-LSTM | 123 | 39 | 12.03 | 10.13 | 25.32 | 7.53 | 32.1 |
| CNN-LSTM | 2026 | 24 | 15.19 | 12.66 | 27.22 | 8.82 | 32.6 |
| ViT | 42 | 17 | 25.32 | 24.05 | 48.73 | 20.05 | 127.7 |
| ViT | 123 | 29 | 22.15 | 23.42 | 51.90 | 19.95 | 129.0 |
| ViT | 2026 | 14 | 21.52 | 19.62 | 47.47 | 15.92 | 128.3 |

## 3. Interpretation

- **Accuracy:** the temporal Vision Transformer roughly doubles top-1 accuracy and macro F1 of the CNN-LSTM, and nearly doubles top-5 accuracy. The gap (≈11 points top-1) is several times the seed-to-seed standard deviation, and the ViT beat the CNN-LSTM on every seed.
- **Speed and size:** the CNN-LSTM is about 4× faster per clip and about 4× smaller. Both are well within real-time limits on the test GPU (a 16-frame clip in 32 ms vs 128 ms).
- **Trade-off:** this matches the exploratory hypothesis in the synopsis — CNN-LSTM gives lower latency, while the temporally modelled ViT gives higher accuracy.
- **Context:** the original WLASL paper (Li et al., 2020) reports 25.97 % top-1 on WLASL-100 for its frame-CNN + RNN baseline (VGG-GRU) using all available videos; this study uses about half the videos (see limitations). Higher literature numbers (I3D ≈ 65.9 %, Pose-TGCN ≈ 55.4 %) come from 3D-CNN or pose-based architectures, which are outside this study's scope.
- **Statistical testing:** with 3 seeds and 158 test clips a formal significance test has very low power; results are reported as mean ± standard deviation, and no p-value is claimed.

## 4. Protocol (what produced these numbers)

| Item | Setting |
|---|---|
| Dataset | WLASL-100 (the 100 most frequent glosses of WLASL v0.3) |
| Videos available | 1,007 of 2,038 annotated instances (rest unavailable from the source) |
| Split | Stratified per class, seed 42: 691 train / 158 val / 158 test; every class in every split |
| Signer independence | **No** — signers overlap between splits (35 train/val, 38 train/test) |
| Input | 16 frames sampled uniformly per clip, resized to 224×224, ImageNet normalisation |
| CNN-LSTM | ResNet-50 (ImageNet, frozen) → 2-layer LSTM (hidden 256) → linear |
| ViT | ViT-B/16 (ImageNet, frozen) → sinusoidal time encoding → 2-layer Transformer encoder (8 heads) → mean pool → linear |
| Training | Identical for both: AdamW lr 3e-4, weight decay 1e-4, dropout 0.3, label smoothing 0.1, batch 32, ≤150 epochs, ReduceLROnPlateau, early stopping (patience 25) |
| Augmentation | 6 cached views per training clip (1 clean + 5 with temporal jitter, 80–100 % crop, horizontal flip), one random view per epoch |
| Checkpoint selection | Highest validation top-1 accuracy (ties → lower validation loss) |
| Latency / FPS | Model forward pass only on GPU, warm-up excluded, CUDA synchronised; excludes video decoding and preprocessing |

## 5. Limitations

1. Only about half of WLASL-100 could be obtained, so absolute accuracies are not directly comparable with papers using the full set.
2. The split is **not signer-independent**; objective 7 (generalisation to unseen signers) could not be evaluated. Results likely overestimate accuracy on new signers.
3. Backbones are frozen ImageNet models; end-to-end fine-tuning was tried and did not improve validation accuracy (see `docs/PROJECT_NOTES.md`).
4. WLASL is **American** Sign Language; the models do not recognise Indian Sign Language.
5. Latency excludes decoding/preprocessing and was measured on one GPU; CPU-only performance will be slower.

## 6. Files

| File | Contents |
|---|---|
| [`results/model_comparison.csv`](results/model_comparison.csv) | Summary table |
| [`results/experimental_results.md`](results/experimental_results.md) / [`.docx`](results/experimental_results.docx) | Auto-generated report |
| [`results/metrics/`](results/metrics/) | Per-run JSON: metrics, per-class scores, confusion matrix, config, hardware |
| [`results/curves/`](results/curves/) | Per-epoch train/val loss and accuracy |
| [`results/figures/`](results/figures/) | Accuracy, F1, latency, FPS, size charts, training curves, confusion matrices |
| [`results/run_log.txt`](results/run_log.txt) | Append-only log of every invocation, including earlier failed/aborted attempts |
| [`datasets/WLASL-100/manifest.csv`](datasets/WLASL-100/manifest.csv) | Exact clip list and split used |
