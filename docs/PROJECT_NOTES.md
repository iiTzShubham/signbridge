# Project Notes — for future reference

A record of how the experiment was set up, what was tried, why decisions were made, and what to do next. Results themselves are in [`../RESULTS.md`](../RESULTS.md).

## 1. Reproducing the results (Windows, NVIDIA GPU)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
# CUDA build of PyTorch first, then the rest of the ML stack
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-ml.txt huggingface_hub
```

1. **Annotations** — download `WLASL_v0.3.json` from the official repo (`start_kit/WLASL_v0.3.json` in https://github.com/dxli94/WLASL) into `datasets/WLASL-100/`.
2. **Videos** — the Kaggle mirror (`risbhawal/wlasl-processed`) is no longer available. The clips were taken from the Hugging Face mirror [Voxel51/WLASL](https://huggingface.co/datasets/Voxel51/WLASL):
   ```powershell
   python experiments/download_wlasl_hf.py experiments/hf_wlasl100_files.json datasets/WLASL-100/videos
   ```
   `hf_wlasl100_files.json` maps each WLASL-100 `video_id` to its file in the mirror (1,007 files, ~480 MB). Review the WLASL C-UDA terms; do not redistribute the videos.
3. **Manifest** — `datasets/WLASL-100/manifest.csv` and `label_map.json` are committed, so the exact split can be reused. To rebuild it:
   ```powershell
   python datasets/build_manifest.py --annotations datasets/WLASL-100/WLASL_v0.3.json --dataset-root datasets/WLASL-100 --classes 100 --seed 42 --split-mode stratified
   python datasets/validate_manifest.py --manifest datasets/WLASL-100/manifest.csv --dataset-root datasets/WLASL-100 --classes 100
   ```
4. **Train + evaluate both models, 3 seeds** (~20 min on an RTX 3060; most of it is the one-time feature extraction):
   ```powershell
   python -m training.compare_models --config experiment_config.json
   python manage.py migrate
   python manage.py import_experiment_results
   python manage.py runserver
   ```
   Dashboard: http://127.0.0.1:8000/research/ · live recognition: http://127.0.0.1:8000/recognize/

Checkpoints (`results/checkpoints/`, 101 MB and 382 MB each) and the feature cache are **not** in git because of GitHub's 100 MB file limit; rerun step 4 to regenerate them.

## 2. Decisions and why

| Decision | Reason |
|---|---|
| Hugging Face mirror for videos | Original YouTube/ASL-site links are mostly dead; the Kaggle copy was removed. |
| Stratified split instead of official split | With only 1,007 clips, the official split left 28 classes with no test clip and 12 with no validation clip, which the pipeline correctly rejects. |
| Not signer-independent | A signer-grouped split was attempted (`--split-mode signer-independent`) but left 17 classes missing from validation and 29 from test. Signer overlap is reported in every metrics file. |
| Frozen ImageNet backbones | Matches the synopsis design (pretrained CNN / ViT feature extractors). Fine-tuning did not help (below). |
| Cached frame features | With frozen backbones, features are identical every epoch. Computing them once cut an epoch from ~90 s to ~0.3 s, enabling 3 seeds and validation-based tuning. Test evaluation still runs the full model on raw video, so latency/FPS are genuine. |
| 16 frames (was 8) | Signs last 1–5 s; 8 frames under-sampled motion. Webcam capture was also lengthened to ~2.5 s (16 frames). |
| lr 3e-4, dropout 0.3, wd 1e-4 | Chosen by a validation-set sweep (`experiments/sweep_heads.py`); same values used for both models so the comparison is fair. |
| Checkpoint by validation top-1 | Validation loss bottomed out early (epoch ~16–26) while validation accuracy kept improving. Decided on validation evidence; the test set was not used for any choice. |
| Batch size 8 (was 1) | The original value was set to save memory on a 16 GB Mac; it only affects raw-video evaluation here. |

## 3. What was tried (validation top-1 unless stated)

| Experiment | CNN-LSTM | ViT | Outcome |
|---|---:|---:|---|
| Original config (8 frames, lr 1e-4, batch 1, no augmentation) | ~0.6 % after 2 epochs, 160 s/epoch | – | Too slow, barely learning; stopped |
| Same with lr 1e-3, 4 data workers | 2.5 % after 3 epochs, 90 s/epoch | – | Still slow; stopped |
| Cached features + 6 augmented views, sweep of lr / dropout / weight decay (cosine schedule) | 15.8–18.4 % | 23.4–30.4 % | Basis of the final setup |
| Same, centre-square crop instead of resize | 18.4–19.0 % | 25.3–27.8 % | No meaningful gain; not adopted |
| End-to-end fine-tuning, backbone lr 3e-5, 20 epochs | 19.6 % | – | Same as frozen, 60 s/epoch; not adopted |
| End-to-end fine-tuning, backbone lr 1e-4, head lr 1e-3 | 1.3 % | – | Collapsed to chance |
| Official pipeline, checkpoint by val loss (test) | 14.1 % test | 22.2 % test | Superseded by val-accuracy selection |
| **Official pipeline, checkpoint by val top-1 (test, final)** | **11.4 % test** | **22.4 % test** | Final results |

Note: one validation clip = 0.63 %, so differences of 1–2 points are within noise.

## 4. Code changes made during the study

- `datasets/build_manifest.py` — new `stratified` split mode.
- `datasets/video_dataset.py` — `augmented_views()` for multi-view augmentation.
- `ml_models/cnn_lstm.py`, `ml_models/vit.py` — `forward()` split into `encode_frames()` + `forward_features()` (same maths, same checkpoints).
- `training/engine.py` — feature caching (`cache_features`, `feature_views`, `head_batch_size`), label smoothing, `checkpoint_metric`, extraction time recorded separately from head training time.
- `recognition/services.py` — Windows fix: temp video file is closed before OpenCV opens it.
- `templates/recognition/recognize.html` — 3-second countdown, ~2.5 s capture (16 frames), top-3 predictions shown; ViT is the default model.

## 5. Future work

1. **Signer-independent evaluation** — obtain more WLASL videos (or the full WLASL-300/2000 for more signers per class) so a signer-grouped split is feasible.
2. **Pose / hand-landmark input** — MediaPipe hand and body keypoints with an LSTM or Transformer are far more data-efficient and robust to background and lighting (e.g. SPOTER, Pose-TGCN ≈ 55 % on WLASL-100).
3. **Video-pretrained backbones** — I3D, VideoMAE or TimeSformer pretrained on Kinetics capture motion directly (I3D ≈ 65.9 % on WLASL-100).
4. **Hand-focused cropping** — detect the signer/hands and crop before feature extraction; hands occupy a small part of a 224×224 frame.
5. **Indian Sign Language** — collect a small ISL dataset with consent (e.g. via a recording page in the app) and retrain, so recognition matches the app's ISL lessons.
6. **Continuous recognition and sentences** — sliding-window recognition plus the "add word to sentence" loop from the system flowchart.
7. **End-to-end latency** — measure decode + preprocessing + model time in the web app, and CPU-only performance, for deployment claims.

## 6. References

1. Li, D., Rodriguez Opazo, C., Yu, X., & Li, H. (2020). Word-level deep sign language recognition from video: A new large-scale dataset and methods comparison. *WACV 2020*, 1459–1469. https://doi.org/10.1109/WACV45572.2020.9093512
2. Dosovitskiy, A., et al. (2021). An image is worth 16×16 words: Transformers for image recognition at scale. *ICLR 2021*. https://arxiv.org/abs/2010.11929
3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR 2016*. https://arxiv.org/abs/1512.03385
4. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.
5. Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS 2017*. https://arxiv.org/abs/1706.03762
6. Carreira, J., & Zisserman, A. (2017). Quo vadis, action recognition? A new model and the Kinetics dataset. *CVPR 2017* (I3D). https://arxiv.org/abs/1705.07750
7. Boháček, M., & Hrúz, M. (2022). Sign pose-based transformer for word-level sign language recognition (SPOTER). *WACV Workshops 2022*.
8. Szegedy, C., et al. (2016). Rethinking the inception architecture for computer vision. *CVPR 2016* (label smoothing). https://arxiv.org/abs/1512.00567
9. WLASL dataset and C-UDA terms: https://dxli94.github.io/WLASL/ · https://github.com/dxli94/WLASL
10. Voxel51 WLASL mirror (video source used): https://huggingface.co/datasets/Voxel51/WLASL
11. PyTorch: https://pytorch.org/docs/ · torchvision models: https://pytorch.org/vision/stable/models.html · OpenCV: https://docs.opencv.org/

Verify page numbers and DOIs against your library before citing in the final paper.
