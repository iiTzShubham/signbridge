# SignBridge — Sign Language Learning and Recognition

SignBridge is a Django web app for learning sign language and recognising signs from video, built for the research project **"A Comparative Study of CNN-LSTM and Vision Transformer Models for Real-Time Sign Language Recognition"** (MCA Sem III, IIMS Chinchwad / SPPU).

It has four parts:

| Page | What it does |
|---|---|
| `/learn/` | Beginner Indian Sign Language (ISL) lessons with ISLRTC videos, search, practice counts and progress |
| `/recognize/` | Recognise one **American Sign Language (ASL) word** from your webcam or an uploaded video, with either model |
| `/translate/` | Small offline English / Hindi / Marathi word helper linking words to ISL lesson videos |
| `/research/` | Dashboard comparing the two models (accuracy, F1, latency, FPS, size, training curves) |

## Results at a glance

Both models were trained on WLASL-100 + ASL Citizen and tested on **11 signers they had never seen** (1,705 clips). Numbers are mean ± std over 3 seeds, measured on an RTX 3060 Laptop GPU.

| | CNN-LSTM | Vision Transformer |
|---|---:|---:|
| Top-1 accuracy | 16.6 ± 1.2 % | **31.4 ± 1.3 %** |
| Top-5 accuracy | 41.9 ± 1.6 % | **60.5 ± 2.0 %** |
| Macro F1-score | 12.9 ± 1.8 % | **28.8 ± 1.7 %** |
| Latency (model only, 16-frame clip) | **31.8 ms** | 143.2 ms |
| Checkpoint size | **101 MB** | 382 MB |

The ViT is about twice as accurate; the CNN-LSTM is about 4.5× faster and 3.8× smaller. Training on WLASL-100 alone (Experiment 1) gave only 6 % / 2 % top-1 on new signers; adding ASL Citizen's webcam signers fixed most of that.

- Full results, both experiments, protocol and limitations: **[RESULTS.md](RESULTS.md)**
- How everything works, decisions, what was tried, reproduction, future work, references: **[docs/PROJECT_NOTES.md](docs/PROJECT_NOTES.md)**

## Quick start (Windows)

Run the web app (lessons, translator, dashboard) — no GPU needed:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
copy .env.example .env
python manage.py migrate
python manage.py import_experiment_results
python manage.py runserver
```

Open <http://127.0.0.1:8000/>. Create an account at **Get started**, or an admin with `python manage.py createsuperuser` (admin at `/admin/`).

### Sign recognition

Recognition also needs the ML packages and the trained checkpoints:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # NVIDIA GPU; drop the URL for CPU-only
python -m pip install -r requirements-ml.txt
```

Checkpoints (`results/checkpoints/`, 101–382 MB each) are too large for GitHub, so they are not in this repository. Re-create them by downloading the data and training (≈2 hours on an RTX 3060) — see [docs/PROJECT_NOTES.md § 2](docs/PROJECT_NOTES.md#2-reproducing-everything-windows-nvidia-gpu). Without them, `/recognize/` explains what is missing instead of guessing.

### Trying `/recognize/`

1. Pick **Vision Transformer** (default, more accurate) or **CNN-LSTM** (faster).
2. **Camera:** click **Start camera**, then **Capture sign**. After the 3-2-1 countdown, sign **one** word for about 2.5 seconds.
   **Upload:** choose a short video of one sign (MP4/WebM, under 50 MB) and click **Recognize this clip**.
3. Read the top answer *and* the top-3 list.

Tips: keep head, hands and upper body in view, use good light and a plain background. The models only know the 100 ASL words listed on the page — not fingerspelled letters (e.g. "C"), sentences, or ISL. Words recognised most reliably for new signers include *dark, son, table, jacket, color, meet, kiss, africa, eat, but, shirt, decide, bed, hat, city*.

## Research pipeline

```powershell
python datasets/build_combined_manifest.py                                     # merge WLASL-100 + ASL Citizen clip lists
python -m training.compare_models --config experiment_config.json             # Experiment 2: train + test both models, 3 seeds
python -m training.compare_models --config experiment_config_wlasl100.json    # Experiment 1: WLASL-100 only
python manage.py import_experiment_results                                     # show Experiment 2 on /research/
```

Each run writes per-seed JSON metrics, `model_comparison*.csv`, an auto-generated report (`experimental_results.md` / `.docx`), charts and training curves to its results folder, and appends to `run_log.txt`. Missing data or packages are reported as "Not measured" — the pipeline never invents numbers. Latency is model-only (decoding excluded); the multi-seed results are mean ± standard deviation without significance claims. Metric definitions: [research/methodology.md](research/methodology.md).

Useful options: `--seeds 42 123 2026`, `--models cnn_lstm`, `--device cpu`, `--epochs 50`.

## Project layout

| Folder | Contents |
|---|---|
| `ml_models/` | The two models (`cnn_lstm.py`, `vit.py`), with plain-language comments |
| `training/` | `compare_models.py` (entry point) and `engine.py` (train/test one model + seed) |
| `datasets/` | Manifest builders and the video loader; videos are downloaded here and git-ignored |
| `results/`, `results_wlasl100/` | Experiment 2 and Experiment 1 outputs |
| `research/` | Dashboard, report generator, results import command |
| `recognition/` | `/recognize/` page and the inference service |
| `experiments/` | Download scripts, tuning sweeps, fine-tuning test, web upload test |
| `learning/`, `translation/`, `accounts/` | Lessons, word helper, sign-up/login |
| `docs/` | Project notes |

## Data and licences

- **WLASL** — C-UDA licence, research use; videos and annotations are not redistributed here. Source: <https://github.com/dxli94/WLASL>; videos used were from the Hugging Face mirror [Voxel51/WLASL](https://huggingface.co/datasets/Voxel51/WLASL).
- **ASL Citizen** — Microsoft Research licence, non-commercial research only; the videos must not be shared or shown in presentations. Source: <https://www.microsoft.com/en-us/research/project/asl-citizen/>.
- **ISL lesson videos** — linked from ISLRTC.

## Configuration

Copy `.env.example` to `.env` and, before any deployment, set a private `SECRET_KEY`, `DEBUG=False` and the correct `ALLOWED_HOSTS`. SQLite (`db.sqlite3`) is used by default. Password-reset emails are printed in the server terminal in development. Never commit `.env`, model weights or datasets.

## App features

- Registration with unique email and password validation; login, logout, password change/reset
- Login-protected dashboard; lessons, practice counts and completion progress
- Lesson content editable in Django Admin
- Signed-in recognition predictions and translator inputs are saved to the user's history; uploaded clips are processed in memory and never saved
- Responsive templates and custom CSS, no frontend build step

## Development checks

```powershell
python manage.py check
python manage.py test
```
