# SignBridge experimental methodology

The comparison runs `cnn_lstm` and `vit_temporal` serially on the same CSV manifest, class-to-index map, seed, and held-out split. The manifest records clip path, gloss, signer ID, split, and frame bounds. The recommended `provided` protocol preserves official WLASL splits where available; missing official partitions require signer-grouped stratification and are recorded as a different protocol. Never compare results across differing manifests as paired results.

Each video is sampled at uniformly spaced frame indices, resized to the configured square image size, converted to RGB, and normalized with ImageNet channel statistics. Both models use ImageNet-pretrained spatial backbones where configured. The CNN-LSTM applies ResNet-50 features per frame and an LSTM over the clip. The temporal ViT applies ViT-B/16 features per frame followed by an explicit Transformer encoder over frame embeddings. By default the pretrained spatial backbones are frozen; temporal layers and classification heads are trained.

Training minimizes cross-entropy using AdamW and a ReduceLROnPlateau schedule. Validation loss selects the saved checkpoint, with early stopping. The test partition is used once after selecting the checkpoint. Reported classification metrics use the same class ordering and macro averaging. Top-5 is omitted only if fewer than five classes exist.

Inference latency is mean per-video model forward-pass time, measured after configurable warm-up batches with accelerator synchronization. The time excludes decoding, frame extraction, and preprocessing; FPS is processed sampled frames divided by total measured model forward time. This is not end-to-end application latency. Peak memory is a sampled process RSS or accelerator allocator/driver measure, not total system usage.

One seed yields preliminary results only. For repeated seeds, tables show arithmetic mean and sample standard deviation. No significance test or p-value is emitted. A future test must respect paired seeds and the independent unit of analysis (typically signer or video, depending on the study design).

## Dataset handling

WLASL annotations and video files are subject to the dataset terms. Obtain videos through the official WLASL maintainers after reviewing the repository's C-UDA terms. Do not redistribute the videos with this project. See the official [WLASL repository](https://github.com/dxli94/WLASL).
