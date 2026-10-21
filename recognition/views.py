import json

from django.contrib import messages
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_http_methods

from .models import RecognitionPrediction
from .services import RecognitionUnavailable, predict_upload


@require_http_methods(["GET", "POST"])
def recognize(request):
    selected_model = request.POST.get("model", "vit_temporal")
    result = None
    error = ""
    if request.method == "POST":
        video = request.FILES.get("video")
        frames = request.FILES.getlist("frames")
        if selected_model not in {"cnn_lstm", "vit_temporal"}:
            error = "Choose one of the available recognition models."
        elif video and video.size > 50 * 1024 * 1024:
            error = "Please choose a video smaller than 50 MB."
        elif video and not (video.content_type or "").startswith("video/"):
            error = "Please choose a video file such as MP4 or WebM."
        else:
            try:
                result = predict_upload(selected_model, video=video, images=frames)
                if request.user.is_authenticated:
                    RecognitionPrediction.objects.create(
                        user=request.user, model_name=result["model"], predicted_sign=result["label"],
                        confidence=result["confidence"], latency_ms=result["latency_ms"],
                    )
            except RecognitionUnavailable as exc:
                error = str(exc)
            except Exception:
                error = "The model could not process this clip. Check the video format and try again."
    if request.headers.get("Accept") == "application/json":
        return JsonResponse({"result": result, "error": error}, status=200 if result else 400)
    history = RecognitionPrediction.objects.filter(user=request.user)[:5] if request.user.is_authenticated else []
    return render(request, "recognition/recognize.html", {"selected_model": selected_model, "result": result, "error": error, "history": history})
