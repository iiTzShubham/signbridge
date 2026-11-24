"""Fingerspelled-letter recognition (ASL A-Z), separate from the 100-word models.

How it works:
  1. MediaPipe Hand Landmarker finds one hand in a camera frame and returns 21 points (wrist, knuckles,
     fingertips).
  2. Still letters (A-Y without J): the points are made position- and size-independent (wrist moved to 0,0,
     scaled by palm length) and mirrored for left hands, so either hand and any distance from the camera
     give similar numbers. A small classifier trained on your own samples picks the letter; scores are
     averaged over the captured frames.
  3. Moving letters (J, Z) are drawn in the air. If the fingertips travel clearly during a capture, a second
     classifier looks at the path of the index and little fingertips plus the handshape at the start and end.
     The "is it moving?" threshold is learned from your own J/Z recordings.

Files (all in trained_models/letters/, git-ignored because they are personal recordings):
  hand_landmarker.task  MediaPipe's hand model, downloaded automatically on first use (~7.5 MB)
  samples.csv           still letters: one row per frame = letter + 63 numbers (21 points x 3)
  motion_samples.csv    J and Z: one row per recording = letter + movement size + path/shape numbers
  letter_model.joblib   still-letter classifier
  motion_model.joblib   J/Z classifier and its movement threshold
"""
import csv
import threading
import time
import urllib.request
from functools import lru_cache

from django.conf import settings

STATIC_LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")   # one still handshape each
MOTION_LETTERS = list("JZ")                          # drawn in the air
ALL_LETTERS = sorted(STATIC_LETTERS + MOTION_LETTERS)
LETTERS = ALL_LETTERS
LETTERS_DIR = settings.BASE_DIR / "trained_models" / "letters"
HAND_MODEL = LETTERS_DIR / "hand_landmarker.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
SAMPLES = LETTERS_DIR / "samples.csv"
MOTION_SAMPLES = LETTERS_DIR / "motion_samples.csv"
CLASSIFIER = LETTERS_DIR / "letter_model.joblib"
MOTION_CLASSIFIER = LETTERS_DIR / "motion_model.joblib"
MIN_SAMPLES_PER_LETTER = 10      # still letters: frames
MIN_MOTION_SAMPLES = 8           # J/Z: recordings
MIN_MOTION_FRAMES = 6            # a J/Z recording needs the hand in at least this many frames
PATH_STEPS = 10                  # J/Z path resampled to this many points
INDEX_TIP, PINKY_TIP, MIDDLE_KNUCKLE = 8, 20, 9
_detector_lock = threading.Lock()  # one MediaPipe detector shared across requests; it is not thread-safe


class LettersUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _detector():
    try:
        from mediapipe.tasks.python import BaseOptions, vision
    except ImportError as exc:
        raise LettersUnavailable("Letter recognition needs MediaPipe. Run: python -m pip install mediapipe") from exc
    if not HAND_MODEL.is_file():
        LETTERS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL)
        except OSError as exc:
            HAND_MODEL.unlink(missing_ok=True)
            raise LettersUnavailable(f"Could not download MediaPipe's hand model. Download {HAND_MODEL_URL} to {HAND_MODEL}.") from exc
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=vision.RunningMode.IMAGE, num_hands=1, min_hand_detection_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def _features(landmarks, handedness):
    """21 MediaPipe points -> 63 numbers that ignore where the hand is and how big it looks."""
    import numpy as np
    points = np.array([(p.x, p.y, p.z) for p in landmarks], dtype=np.float32)
    points -= points[0]                                     # wrist at the origin
    if handedness == "Left":
        points[:, 0] *= -1                                  # mirror so left and right hands match
    palm = float(np.linalg.norm(points[9, :2])) or 1.0      # wrist -> middle-finger knuckle
    return (points / palm).flatten()


def _detect(image_bgr):
    """One frame -> (63 shape numbers, 21x2 pixel positions, "Left"/"Right"), or None if no hand.
    Pixel positions are NOT mirrored: MediaPipe's left/right label can flip between frames, so the
    movement code mirrors a whole recording once (see _motion_features) instead of frame by frame."""
    import cv2
    import mediapipe as mp
    import numpy as np
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    with _detector_lock:
        result = _detector().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.hand_landmarks:
        return None
    landmarks, handedness = result.hand_landmarks[0], result.handedness[0][0].category_name
    height, width = image_bgr.shape[:2]
    pixels = np.array([(p.x * width, p.y * height) for p in landmarks], dtype=np.float32)
    return _features(landmarks, handedness), pixels, handedness


def landmarks_from_image(image_bgr):
    """The 63-number shape vector for the most confident hand, or None if no hand is visible."""
    found = _detect(image_bgr)
    return None if found is None else found[0]


def _movement(pixels):
    """How far the index and little fingertips travel during a capture, in palm lengths."""
    import numpy as np
    palm = float(np.median([np.linalg.norm(p[MIDDLE_KNUCKLE] - p[0]) for p in pixels])) or 1.0
    spans = [np.linalg.norm(np.ptp(np.array([p[tip] for p in pixels]), axis=0)) for tip in (INDEX_TIP, PINKY_TIP)]
    return float(max(spans)) / palm, palm


def _motion_features(detections):
    """A J/Z recording -> fixed-length numbers: fingertip paths (relative to the start, in palm lengths)
    resampled to PATH_STEPS points, plus the average handshape in the first and last third."""
    import numpy as np
    shapes = np.array([shape for shape, _, _ in detections])
    pixels = [p for _, p, _ in detections]
    movement, palm = _movement(pixels)
    start = pixels[0][0]
    tips = np.array([[(p[INDEX_TIP] - start) / palm, (p[PINKY_TIP] - start) / palm] for p in pixels])  # T x 2 x 2
    if sum(hand == "Left" for _, _, hand in detections) > len(detections) / 2:
        tips[:, :, 0] *= -1   # left hand: mirror the whole path once so left- and right-handed J/Z match
    steps = np.linspace(0, len(pixels) - 1, PATH_STEPS)
    path = np.stack([np.interp(steps, np.arange(len(pixels)), tips[:, t, c]) for t in range(2) for c in range(2)], axis=1)
    path -= path[0]
    third = max(1, len(shapes) // 3)
    return movement, np.concatenate([path.flatten(), shapes[:third].mean(axis=0), shapes[-third:].mean(axis=0)])


def decode_uploads(uploads):
    """JPEG/PNG uploads (camera frames or a photo) -> list of BGR images."""
    import cv2
    import numpy as np
    images = []
    for upload in uploads:
        image = cv2.imdecode(np.frombuffer(upload.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is not None:
            images.append(image)
    return images


# ---------- recording ----------

def _count(path, counts):
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.reader(handle):
                if row and row[0] in counts:
                    counts[row[0]] += 1


def sample_counts():
    counts = {letter: 0 for letter in ALL_LETTERS}
    _count(SAMPLES, counts)
    _count(MOTION_SAMPLES, counts)
    return counts


def _append(path, rows):
    LETTERS_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)


def add_samples(letter, images):
    """Still letter: one row per frame with a hand. J/Z: one row for the whole recording.
    Returns a short message for the recording page."""
    if letter not in ALL_LETTERS:
        raise LettersUnavailable("Choose a letter from A to Z.")
    detections = [found for found in map(_detect, images) if found is not None]
    if letter in MOTION_LETTERS:
        if len(detections) < MIN_MOTION_FRAMES:
            return f"Hand found in only {len(detections)} of {len(images)} frames, so this {letter} was not saved. Keep your hand in view while drawing."
        movement, features = _motion_features(detections)
        _append(MOTION_SAMPLES, [[letter, f"{movement:.4f}", *(f"{v:.5f}" for v in features)]])
        return f"Saved 1 recording of {letter} (hand in {len(detections)}/{len(images)} frames, moved {movement:.1f} palm lengths)."
    if detections:
        _append(SAMPLES, [[letter, *(f"{v:.5f}" for v in shape)] for shape, _, _ in detections])
    message = f"Saved {len(detections)} of {len(images)} frames for {letter}."
    return message if detections else message + " No hand was found - check the light and keep your hand in frame."


def delete_samples(letter):
    path = MOTION_SAMPLES if letter in MOTION_LETTERS else SAMPLES
    if not path.is_file():
        return
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.reader(handle) if row and row[0] != letter]
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)


# ---------- training ----------

def _svm():
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    # SVM on scaled numbers; the calibration wrapper turns its scores into percentages.
    return make_pipeline(StandardScaler(), CalibratedClassifierCV(SVC(C=10, gamma="scale"), cv=3, ensemble=False))


def _fit(x, y):
    """Fit the classifier and return it with a cross-validated accuracy estimate."""
    import numpy as np
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    folds = min(5, min(np.unique(y, return_counts=True)[1]))
    scores = cross_val_score(_svm(), x, y, cv=StratifiedKFold(folds, shuffle=True, random_state=42))
    return _svm().fit(x, y), float(scores.mean()) * 100


def _read(path, letters, skip=0):
    import numpy as np
    labels, extras, features = [], [], []
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.reader(handle):
                if row and row[0] in letters:
                    labels.append(row[0])
                    extras.append([float(v) for v in row[1:1 + skip]])
                    features.append([float(v) for v in row[1 + skip:]])
    return np.array(features, dtype=np.float32), np.array(labels), np.array(extras, dtype=np.float32)


def train():
    """Train the still-letter model and (if J and Z are recorded) the J/Z model."""
    import joblib
    import numpy as np
    counts = sample_counts()
    still = [l for l in STATIC_LETTERS if counts[l] >= MIN_SAMPLES_PER_LETTER]
    moving = [l for l in MOTION_LETTERS if counts[l] >= MIN_MOTION_SAMPLES]
    if len(still) < 2 and len(moving) < 2:
        raise LettersUnavailable(f"Record at least {MIN_SAMPLES_PER_LETTER} samples for two or more letters "
                                 f"(or {MIN_MOTION_SAMPLES} recordings each of J and Z) first.")
    LETTERS_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = {"letters": [], "missing": [], "samples": 0, "cv_accuracy": None, "motion_accuracy": None}
    if len(still) >= 2:
        x, y, _ = _read(SAMPLES, still)
        model, report["cv_accuracy"] = _fit(x, y)
        joblib.dump({"model": model, "letters": still}, CLASSIFIER)
        report["letters"] += still
        report["samples"] += len(y)
    if len(moving) == 2:
        x, y, movement = _read(MOTION_SAMPLES, moving, skip=1)
        model, report["motion_accuracy"] = _fit(x, y)
        # A capture counts as "moving" if the fingertips travel at least half as far as in a typical
        # (20th-percentile) J/Z recording, kept between 0.35 and 1.5 palm lengths.
        threshold = float(np.clip(0.5 * np.percentile(movement[:, 0], 20), 0.35, 1.5))
        joblib.dump({"model": model, "letters": moving, "threshold": threshold}, MOTION_CLASSIFIER)
        report["letters"] += moving
        report["samples"] += len(y)
        report["motion_threshold"] = threshold
    elif MOTION_CLASSIFIER.is_file() and len(moving) < 2:
        MOTION_CLASSIFIER.unlink()   # J/Z samples were deleted; do not keep an outdated J/Z model
    report["missing"] = [l for l in ALL_LETTERS if l not in report["letters"]]
    report["letters"] = sorted(report["letters"])
    report["train_seconds"] = time.perf_counter() - started
    _classifier.cache_clear()
    _motion_classifier.cache_clear()
    return report


# ---------- prediction ----------

@lru_cache(maxsize=1)
def _classifier():
    import joblib
    return joblib.load(CLASSIFIER) if CLASSIFIER.is_file() else None


@lru_cache(maxsize=1)
def _motion_classifier():
    import joblib
    return joblib.load(MOTION_CLASSIFIER) if MOTION_CLASSIFIER.is_file() else None


def model_ready():
    return CLASSIFIER.is_file() or MOTION_CLASSIFIER.is_file()


def _result(probabilities, classes, started, frames, kind, movement):
    import numpy as np
    order = np.argsort(probabilities)[::-1]
    classes = [str(name) for name in classes]
    return {
        "label": classes[order[0]],
        "confidence": float(probabilities[order[0]]) * 100,
        "latency_ms": (time.perf_counter() - started) * 1000,
        "model": "letters",
        "device": f"cpu · {kind} · moved {movement:.1f} palms",
        "frames_with_hand": frames,
        "top_predictions": [{"label": classes[i], "confidence": float(probabilities[i]) * 100} for i in order[:3]],
    }


def predict_letter(images):
    """Moving hand -> J/Z model; still hand -> still-letter model (scores averaged over the frames)."""
    import numpy as np
    if not images:
        raise LettersUnavailable("Capture a few camera frames or choose a photo of your hand.")
    still_bundle, motion_bundle = _classifier(), _motion_classifier()
    if still_bundle is None and motion_bundle is None:
        raise LettersUnavailable("No letter model yet. Record your letters on the 'Record letters' page and press Train.")
    started = time.perf_counter()
    detections = [found for found in map(_detect, images) if found is not None]
    if not detections:
        raise LettersUnavailable("No hand was found. Hold one hand clearly in front of the camera, in good light, and try again.")
    movement = _movement([p for _, p, _ in detections])[0] if len(detections) > 1 else 0.0
    moving = motion_bundle is not None and len(detections) >= MIN_MOTION_FRAMES and movement >= motion_bundle["threshold"]
    if moving or still_bundle is None:
        if motion_bundle is None or len(detections) < MIN_MOTION_FRAMES:
            raise LettersUnavailable("Only J and Z are trained so far, and they need a 2-second camera capture while you draw the letter.")
        model = motion_bundle["model"]
        probabilities = model.predict_proba(_motion_features(detections)[1][None, :].astype(np.float32))[0]
        return _result(probabilities, model.classes_, started, len(detections), "movement (J/Z)", movement)
    model = still_bundle["model"]
    probabilities = model.predict_proba(np.array([shape for shape, _, _ in detections], dtype=np.float32)).mean(axis=0)
    return _result(probabilities, model.classes_, started, len(detections), "handshape", movement)
