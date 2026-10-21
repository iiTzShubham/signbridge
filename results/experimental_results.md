# A Comparative Study of CNN-LSTM and Vision Transformer Models for Real-Time Sign Language Recognition — Experiment 2 (WLASL-100 + ASL Citizen)

## 1. Experimental Setup

- Dataset: WLASL-100 + ASL Citizen (100 words)
- Classes configured: 100
- Frames per clip: 16; image size: 224
- Configured split: combined: WLASL-100 stratified train/val + ASL Citizen official signer-independent train/val/test; WLASL test kept as test_wlasl; seed(s): [42, 123, 2026]
- Training settings: epochs=150, batch size=8, learning rate=0.0003, pretrained=True
- Environment: {"platform": "Windows-11-10.0.26220-SP0", "machine": "AMD64", "processor": "AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD", "cpu_count": 16, "memory_bytes": null, "hardware_model": null, "gpu": "NVIDIA GeForce RTX 3060 Laptop GPU", "accelerator": "cuda"}; software: {"python": "3.12.3", "Django": "5.2.18", "torch": "2.11.0+cu128", "torchvision": "0.26.0+cu128", "opencv-python-headless": "4.14.0", "numpy": "2.5.2", "pandas": "3.0.6", "scikit-learn": "1.9.1", "matplotlib": "3.11.2", "python-docx": "1.2.0", "scipy": "1.18.1"}

The saved experiment configuration is `experiment_config.json`. Each complete JSON record contains the actual split counts, test-video count, signer overlap, device, training duration, and model resource measurements.

## 2. Results

| Evaluation metric | CNN-LSTM | Vision Transformer | Better result |
|---|---:|---:|---|
| Top-1 accuracy (%) | 16.598 ± 1.177 (n=3) | 31.398 ± 1.301 (n=3) | Vision Transformer |
| Top-5 accuracy (%) | 41.877 ± 1.575 (n=3) | 60.489 ± 2.042 (n=3) | Vision Transformer |
| Macro precision (%) | 15.975 ± 2.143 (n=3) | 32.861 ± 2.587 (n=3) | Vision Transformer |
| Macro recall (%) | 13.365 ± 1.623 (n=3) | 29.331 ± 1.337 (n=3) | Vision Transformer |
| Macro F1-score (%) | 12.869 ± 1.756 (n=3) | 28.843 ± 1.677 (n=3) | Vision Transformer |
| Average inference latency (ms/clip) | 31.805 ± 0.158 (n=3) | 143.216 ± 3.878 (n=3) | CNN-LSTM |
| FPS (sampled frames/s) | 503.072 ± 2.491 (n=3) | 111.774 ± 3.020 (n=3) | CNN-LSTM |
| Model checkpoint size (MB) | 101.098 ± 0.001 (n=3) | 381.743 ± 0.000 (n=3) | CNN-LSTM |
| Training duration (s) | 52.766 ± 4.591 (n=3) | 99.647 ± 5.737 (n=3) | CNN-LSTM |

Completed runs: CNN-LSTM 3; Vision Transformer 3.

Metrics are means across successful seeds when there is more than one run; the accompanying `±` is sample standard deviation. A single run is preliminary and does not estimate run-to-run variability.

### Additional test set: `test_wlasl` (158 clips)

| Evaluation metric | CNN-LSTM | Vision Transformer | Better result |
|---|---:|---:|---|
| Top-1 accuracy (%) | 21.097 ± 2.923 (n=3) | 34.388 ± 1.462 (n=3) | Vision Transformer |
| Top-5 accuracy (%) | 47.046 ± 2.991 (n=3) | 61.814 ± 2.558 (n=3) | Vision Transformer |
| Macro precision (%) | 19.117 ± 1.592 (n=3) | 31.417 ± 1.956 (n=3) | Vision Transformer |
| Macro recall (%) | 20.667 ± 3.686 (n=3) | 36.000 ± 1.803 (n=3) | Vision Transformer |
| Macro F1-score (%) | 18.010 ± 2.474 (n=3) | 31.139 ± 1.614 (n=3) | Vision Transformer |

## 3. Comparative Analysis

The table reports measured values for both architectures. Per-metric direction is used for the ‘Better result’ column; these measurements do not establish a universal deployment winner. Latency and FPS here time the model forward pass on the recorded device and exclude video decoding and preprocessing.

## 4. Discussion

Differences may be associated with spatial feature extraction, temporal aggregation, model capacity, pretrained weights, and the dataset's signer and class distribution. These are possible explanations, not findings, unless supported by completed runs and the recorded protocol. The result files and training curves are the evidence for this implementation.

## 5. Hypothesis Evaluation

Multiple runs are available for both models. Mean and standard deviation are reported. No p-value is produced by this pipeline; a statistical test should only be added after checking paired seeds, distributional assumptions, and the selected unit of analysis.

## 6. Limitations

- Dataset availability, split integrity, class balance, and signer overlap must be verified from the recorded manifest and run metadata.
- Results are limited to the configured WLASL-100 + ASL Citizen (100 words) data, preprocessing, architecture settings, seeds, and available hardware.
- The latency measure excludes decoding and preprocessing and should not be interpreted as end-to-end real-time performance.
- Statistical significance is not reported unless an appropriate analysis is separately implemented and justified.

## 7. Conclusion

No research conclusion can be drawn until both models complete evaluation on the shared held-out test split. This report intentionally leaves unavailable metrics as ‘Not measured’ and does not import published benchmark values into project results.

## Published literature (separate from project measurements)

No published benchmark numbers are included in this report. Add literature results only with their original citation, dataset, and evaluation conditions; they must remain distinct from these implementation measurements.

## Figures

![Accuracy Comparison](figures/accuracy_comparison.png)

*Figure: Accuracy Comparison generated from completed experiment records.*

![Cnn Lstm Confusion Matrix](figures/cnn_lstm_confusion_matrix.png)

*Figure: Cnn Lstm Confusion Matrix generated from completed experiment records.*

![F1 Comparison](figures/f1_comparison.png)

*Figure: F1 Comparison generated from completed experiment records.*

![Fps Comparison](figures/fps_comparison.png)

*Figure: Fps Comparison generated from completed experiment records.*

![Latency Comparison](figures/latency_comparison.png)

*Figure: Latency Comparison generated from completed experiment records.*

![Model Size Comparison](figures/model_size_comparison.png)

*Figure: Model Size Comparison generated from completed experiment records.*

![Training Curves](figures/training_curves.png)

*Figure: Training Curves generated from completed experiment records.*

![Validation Performance](figures/validation_performance.png)

*Figure: Validation Performance generated from completed experiment records.*

![Vision Transformer Confusion Matrix](figures/vision_transformer_confusion_matrix.png)

*Figure: Vision Transformer Confusion Matrix generated from completed experiment records.*

