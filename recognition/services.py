"""Load trained recognition checkpoints on demand; never return a demo prediction."""
import json
import tempfile
import time
import copy
from functools import lru_cache
from pathlib import Path

from django.conf import settings


class RecognitionUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=2)
def _load_model(model_name):
    root = settings.BASE_DIR
    config_path = root / "experiment_config.json"
    if not config_path.is_file():
        raise RecognitionUnavailable("Recognition is not configured yet. Add experiment_config.json and a trained checkpoint.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seed = int(config.get("seed", 42))
    checkpoint = root / config.get("results_dir", "results") / "checkpoints" / model_name / f"seed_{seed}.pth"
    label_file = (root / config.get("manifest_file", "datasets/WLASL-100/manifest.csv")).with_name("label_map.json")
    if not checkpoint.is_file():
        raise RecognitionUnavailable(f"The {model_name.replace('_', ' ').upper()} model has no trained checkpoint yet. Run the paired training workflow first.")
    if not label_file.is_file():
        raise RecognitionUnavailable("The model checkpoint is present, but its sign label map is missing. Rebuild the dataset manifest first.")
    try:
        import torch
        from ml_models.factory import build_model
        from training.utils import select_device, synchronize
    except ImportError as exc:
        raise RecognitionUnavailable("Recognition libraries are not installed. Run: python -m pip install -r requirements-ml.txt") from exc

    labels_by_name = json.loads(label_file.read_text(encoding="utf-8"))
    labels = [name for name, _index in sorted(labels_by_name.items(), key=lambda pair: int(pair[1]))]
    device = select_device("auto")
    model_config = dict(config)
    model_config["pretrained"] = False  # The checkpoint already contains the trained model weights.
    model = build_model(model_name, len(labels), model_config).to(device)
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:  # Older PyTorch versions do not expose weights_only.
        state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state.get("state_dict", state))
    model.eval()
    return model, labels, device, config, torch, synchronize


def _decode_video(upload, frame_count, image_size):
    import cv2
    import numpy as np
    suffix = Path(upload.name).suffix or ".mp4"
    # Windows cannot reopen a file that is still open, so close it before OpenCV reads it.
    with tempfile.NamedTemporaryFile(suffix=suffix, delete_on_close=False) as temp:
        for chunk in upload.chunks():
            temp.write(chunk)
        temp.close()
        capture = cv2.VideoCapture(temp.name)
        try:
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if total < 1:
                raise RecognitionUnavailable("This video could not be opened. Try an MP4 or use the camera capture.")
            indices = np.linspace(0, total - 1, num=frame_count, dtype=int)
            decoded = []
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = capture.read()
                if not ok:
                    continue
                decoded.append(cv2.cvtColor(cv2.resize(frame, (image_size, image_size)), cv2.COLOR_BGR2RGB))
            if not decoded:
                raise RecognitionUnavailable("No frames could be read from this video.")
            while len(decoded) < frame_count:
                decoded.append(decoded[-1].copy())
            return decoded[:frame_count]
        finally:
            capture.release()


def _decode_images(uploads, frame_count, image_size):
    import cv2
    import numpy as np
    decoded = []
    for upload in uploads:
        content = upload.read()
        image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is not None:
            decoded.append(cv2.cvtColor(cv2.resize(image, (image_size, image_size)), cv2.COLOR_BGR2RGB))
    if not decoded:
        raise RecognitionUnavailable("The camera frames could not be read. Please try capturing again.")
    indices = np.linspace(0, len(decoded) - 1, num=frame_count, dtype=int)
    return [decoded[index] for index in indices]


def predict_upload(model_name, video=None, images=None):
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise RecognitionUnavailable("Recognition libraries are not installed. Run: python -m pip install -r requirements-ml.txt") from exc
    model, labels, device, config, torch, synchronize = _load_model(model_name)
    frame_count, image_size = int(config.get("num_frames", 8)), int(config.get("image_size", 224))
    if video:
        frames = _decode_video(video, frame_count, image_size)
    elif images:
        frames = _decode_images(images, frame_count, image_size)
    else:
        raise RecognitionUnavailable("Choose a video clip or capture a few frames with your camera.")
    array = np.asarray(frames, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2)
    mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    tensor = ((tensor - mean) / std).unsqueeze(0).to(device)
    synchronize(device)
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            probabilities = torch.softmax(model(tensor), dim=1)[0]
    except RuntimeError as exc:
        message = str(exc).lower()
        if getattr(device, "type", "") != "mps" or not any(term in message for term in ("mps", "out of memory", "not implemented")):
            raise
        # Retry inference on CPU if an MPS kernel or memory limit prevents this clip.
        device = torch.device("cpu")
        model = copy.deepcopy(model).to(device)
        tensor = tensor.to(device)
        synchronize = lambda _device: None
        with torch.inference_mode():
            probabilities = torch.softmax(model(tensor), dim=1)[0]
    synchronize(device)
    latency_ms = (time.perf_counter() - started) * 1000
    confidence, index = probabilities.max(dim=0)
    probabilities = probabilities.detach().cpu()
    return {
        "label": labels[int(index)],
        "confidence": float(confidence.detach().cpu().item()) * 100,
        "latency_ms": latency_ms,
        "model": model_name,
        "device": str(device),
        "top_predictions": [{"label": labels[i], "confidence": float(probabilities[i].item()) * 100} for i in torch.topk(probabilities, k=min(3, len(labels))).indices.tolist()],
    }
