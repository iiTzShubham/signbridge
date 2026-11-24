"""The /recognize/ page: shows the camera/upload form and, on POST, sends the clip or camera frames
to recognition/services.py (100 ASL words) or recognition/letters.py (fingerspelled letters) and returns
the prediction (as HTML, or JSON for the camera script). /recognize/letters/ records and trains letters.
"""
import json

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_http_methods

from . import letters
from .models import RecognitionPrediction
from .services import RecognitionUnavailable, predict_upload

WORD_MODELS = {"cnn_lstm", "vit_temporal"}


def _vocabulary():
    """The words the trained model can recognise (from label_map.json next to the manifest)."""
    try:
        config = json.loads((settings.BASE_DIR / "experiment_config.json").read_text(encoding="utf-8"))
        label_map = (settings.BASE_DIR / config["manifest_file"]).with_name("label_map.json")
        return sorted(json.loads(label_map.read_text(encoding="utf-8")))
    except (OSError, KeyError, ValueError):
        return []


def _predict(request, selected_model):
    if selected_model == "letters":
        image = request.FILES.get("image")
        if image and not (image.content_type or "").startswith("image/"):
            raise RecognitionUnavailable("Please choose a photo (JPEG or PNG) of your hand.")
        uploads = [image] if image else request.FILES.getlist("frames")
        try:
            return letters.predict_letter(letters.decode_uploads(uploads))
        except letters.LettersUnavailable as exc:
            raise RecognitionUnavailable(str(exc)) from exc
    video = request.FILES.get("video")
    if video and video.size > 50 * 1024 * 1024:
        raise RecognitionUnavailable("Please choose a video smaller than 50 MB.")
    if video and not (video.content_type or "").startswith("video/"):
        raise RecognitionUnavailable("Please choose a video file such as MP4 or WebM.")
    return predict_upload(selected_model, video=video, images=request.FILES.getlist("frames"))


@require_http_methods(["GET", "POST"])
def recognize(request):
    selected_model = request.POST.get("model") or request.GET.get("mode") or "vit_temporal"
    if selected_model == "words":
        selected_model = "vit_temporal"
    result = None
    error = ""
    if request.method == "POST":
        if selected_model not in WORD_MODELS | {"letters"}:
            error = "Choose one of the available recognition models."
        else:
            try:
                result = _predict(request, selected_model)
                if request.user.is_authenticated:
                    RecognitionPrediction.objects.create(
                        user=request.user, model_name=result["model"], predicted_sign=result["label"],
                        confidence=result["confidence"], latency_ms=result["latency_ms"],
                    )
            except RecognitionUnavailable as exc:
                error = str(exc)
            except Exception:
                error = "The model could not process this input. Check the file format and try again."
    if request.headers.get("Accept") == "application/json":
        return JsonResponse({"result": result, "error": error}, status=200 if result else 400)
    history = RecognitionPrediction.objects.filter(user=request.user)[:5] if request.user.is_authenticated else []
    return render(request, "recognition/recognize.html", {
        "selected_model": selected_model, "mode": "letters" if selected_model == "letters" else "words",
        "result": result, "error": error, "history": history, "vocabulary": _vocabulary(),
        "letters": letters.ALL_LETTERS, "motion_letters": letters.MOTION_LETTERS, "letter_model_ready": letters.model_ready(),
    })


@require_http_methods(["GET", "POST"])
def record_letters(request):
    """Record your own fingerspelling samples (JSON actions: add / delete / train)."""
    if request.method == "GET":
        counts = letters.sample_counts()
        tiles = [{"letter": l, "count": n, "target": 20 if l in letters.MOTION_LETTERS else 40, "motion": l in letters.MOTION_LETTERS}
                 for l, n in counts.items()]
        return render(request, "recognition/record_letters.html", {
            "tiles": tiles, "minimum": letters.MIN_SAMPLES_PER_LETTER, "motion_minimum": letters.MIN_MOTION_SAMPLES,
            "model_ready": letters.model_ready(),
        })
    action, letter = request.POST.get("action"), request.POST.get("letter", "")
    try:
        if action == "add":
            payload = {"message": letters.add_samples(letter, letters.decode_uploads(request.FILES.getlist("frames")))}
        elif action == "delete":
            letters.delete_samples(letter)
            payload = {"message": f"Deleted all samples for {letter}."}
        elif action == "train":
            report = letters.train()
            parts = [f"Trained {len(report['letters'])} letters on {report['samples']} samples."]
            if report["cv_accuracy"] is not None:
                parts.append(f"Still letters: about {report['cv_accuracy']:.0f}% accurate on your own signing.")
            if report["motion_accuracy"] is not None:
                parts.append(f"J/Z: about {report['motion_accuracy']:.0f}% (a capture counts as moving above {report['motion_threshold']:.1f} palm lengths).")
            if report["missing"]:
                parts.append(f"Not yet recorded enough: {', '.join(report['missing'])}.")
            payload = {"message": " ".join(parts), "report": report}
        else:
            return JsonResponse({"error": "Unknown action."}, status=400)
    except letters.LettersUnavailable as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    payload["counts"] = letters.sample_counts()
    return JsonResponse(payload)
