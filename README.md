# SignBridge — Sign Language Learning and Recognition

SignBridge is a Django foundation for a sign language learning, recognition, and translation platform. It provides the project setup, SQLite development database, account registration and login, password reset flow, responsive pages, and a beginner Indian Sign Language (ISL) lesson library. The research module includes paired CNN-LSTM and temporal Vision Transformer training/evaluation, result reporting, and a Django comparison dashboard. It reports no model performance until actual completed runs are recorded.

## Results at a glance

Measured on the WLASL-100 test split (mean ± std over 3 seeds, RTX 3060 Laptop GPU):

| | CNN-LSTM | Vision Transformer |
|---|---:|---:|
| Top-1 accuracy | 11.4 ± 1.3 % | **22.4 ± 2.4 %** |
| Top-5 accuracy | 27.4 ± 2.2 % | **49.4 ± 2.3 %** |
| Latency (model only) | **31.7 ms/clip** | 128.3 ms/clip |
| Checkpoint size | **101 MB** | 382 MB |

Full results, protocol and limitations: [RESULTS.md](RESULTS.md). Decisions, experiments tried, reproduction steps, future work and references: [docs/PROJECT_NOTES.md](docs/PROJECT_NOTES.md).

## Quick start (Windows)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
copy .env.example .env
python manage.py migrate
python manage.py runserver
```

For training and live recognition, also install the ML stack (see [docs/PROJECT_NOTES.md](docs/PROJECT_NOTES.md#1-reproducing-the-results-windows-nvidia-gpu)). Trained checkpoints are not stored in git; regenerate them with `python -m training.compare_models --config experiment_config.json`.

## Requirements

- Python 3.10 or later (Python 3.12 recommended)
- Git (optional for local development)
- No Node.js build step

## Setup

```bash
git clone <repository-url>
cd signbridge
# On macOS, use the versioned executable so the system Python isn't selected.
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
python manage.py migrate
python manage.py runserver
```

If `python3.12` is not installed, install Python 3.12 from [python.org](https://www.python.org/downloads/) first. Django 5.2 does not support Python 3.9. After creating the environment, select the workspace's `.venv` interpreter in your IDE. If an older `.venv` already exists, remove it before recreating it with Python 3.12.

Open <http://127.0.0.1:8000/>. Create an account at **Get started** or create an admin with `python manage.py createsuperuser` and open <http://127.0.0.1:8000/admin/>.

On Windows, activate the virtual environment with `.venv\Scripts\activate`.

## Configuration

Copy `.env.example` to `.env` before deployment and set a unique, private `SECRET_KEY`, `DEBUG=False`, and the appropriate `ALLOWED_HOSTS`. The settings loader reads simple `KEY=value` lines and does not require an extra package. SQLite is used by default at `db.sqlite3`. `TRANSLATION_API_KEY` and `MODEL_PATH` are reserved for later modules. Never commit `.env`, real API keys, model weights, or personal datasets.

## Phase 1 features

- Django project with SQLite and Django's secure built-in authentication
- Registration with username, name, unique email, password validation, and hashed passwords
- Login, logout (CSRF-protected POST), password change/reset foundation
- Public landing page and a login-protected dashboard
- Public beginner lesson library at `/learn/`, with search and category filters
- Starter ISL lessons for the alphabet, greetings, self-introduction, and common words, using ISLRTC instructional videos
- Signed-in lesson views, practice counts, and completion progress
- Lesson content managed in the database and editable in Django Admin
- Responsive templates and custom CSS; no frontend build tool required
- Django admin enabled

Password reset uses Django's console email backend in development. Reset messages are printed in the server terminal. Configure a real email backend before production use.

## Planned modules

The project is intended to grow into independent Django apps for translation, recognition, history, and API access. PyTorch architectures, data loaders, training/evaluation scripts, and actual experiment records are kept separate from web views. Recognition results and performance metrics must come from a loaded model and recorded evaluation; the interface must never invent them.

## Research experiments

Research scripts use an optional dependency file so the Django learning site does not require PyTorch. Activate `.venv` and install the ML stack when ready:

```bash
python -m pip install -r requirements-ml.txt
```

The default protocol expects WLASL-100 metadata at `datasets/WLASL-100/WLASL_v0.3.json` and authorized clips at `datasets/WLASL-100/videos/<video_id>.mp4`. WLASL access and use are governed by the official [WLASL repository and C-UDA terms](https://github.com/dxli94/WLASL); request/access the videos from the maintainers and do not redistribute them. The repository does not include WLASL videos or annotations.

Build and validate the local manifest:

```bash
python datasets/build_manifest.py --annotations datasets/WLASL-100/WLASL_v0.3.json --dataset-root datasets/WLASL-100 --classes 100 --seed 42 --split-mode provided
python datasets/validate_manifest.py --manifest datasets/WLASL-100/manifest.csv --dataset-root datasets/WLASL-100 --classes 100
```

If you create a signer-independent split, use `--split-mode signer-independent`; it requires signer IDs and every class in all three partitions. Record the split protocol and do not compare results from different manifests as paired experiments. Review and update `experiment_config.json` before running. The machine checked during development is an Apple Silicon M4 with 16 GiB RAM; `device: auto` prefers CUDA/MPS when available and otherwise CPU. The configured batch size is 1 to bound memory. Pretrained torchvision weights may need to download on the first run.

Run both models on the same manifest, split, and configured seeds:

```bash
python -m training.compare_models --config experiment_config.json
```

Adjust resources or seeds without editing the file:

```bash
python -m training.compare_models --config experiment_config.json --seeds 42 123 2026 --batch-size 1 --num-frames 8 --device auto
```

The runner records complete metrics only after the validation-selected checkpoint is evaluated on the held-out test split. It creates `results/model_comparison.csv`, `experimental_results.md`, editable `experimental_results.docx` (when `python-docx` is installed), per-seed JSON metrics, configuration, status, and an append-only run log. Figures are emitted only from successful measured runs. Missing dependencies or data are recorded in the report and log as “Not measured”. The comparison dashboard is at `/research/`; import run records after training with:

```bash
python manage.py migrate
python manage.py import_experiment_results
```

For methodology and metric definitions see [`research/methodology.md`](research/methodology.md). Inference latency is model-only and excludes decoding and preprocessing, so it is not end-to-end web application latency. Multiple seeds produce mean and standard deviation; the pipeline does not make significance claims.

## Recognition and text-to-sign tools

- `/recognize/` supports browser camera capture and video upload. It makes predictions only when a trained checkpoint exists at `results/checkpoints/<model>/seed_<seed>.pth`; otherwise it explains what is missing. The current research config uses WLASL, an American Sign Language dataset, so those weights are not suitable for ISL recognition. ISL prediction requires an ISL-trained checkpoint and label map.
- `/translate/` provides a small offline English/Hindi/Marathi word helper and links words to related ISLRTC lesson videos or the ISL alphabet visual guide. It is a vocabulary learning aid, not a fluent translator; ISL has its own grammar and meanings depend on context.
- Beginner lessons at `/learn/` include ISLRTC videos and visual references. Signed-in recognition predictions and translator inputs are saved to the account history tables; uploaded clips are processed in memory and are not saved as files.

## Development checks

Once Python and dependencies are available, run:

```bash
python manage.py check
python manage.py migrate
python manage.py runserver
```

Authentication, app, and API tests will be added with their corresponding modules. They have not been run in this environment because the current system Python cannot start until Command Line Tools are installed.
