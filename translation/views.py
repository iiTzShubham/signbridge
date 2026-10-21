from django.shortcuts import render
from django.views.decorators.http import require_http_methods

from learning.models import Lesson
from .models import Sign, TranslationHistory
from .services import LANGUAGE_NAMES, sign_glosses, translate_text


@require_http_methods(["GET", "POST"])
def translator(request):
    text = ""
    source_language, target_language = "en", "hi"
    translated_text = ""
    cards = []
    error = ""
    if request.method == "POST":
        text = request.POST.get("text", "").strip()[:1000]
        source_language = request.POST.get("source_language", "en")
        target_language = request.POST.get("target_language", "hi")
        if not text:
            error = "Write a short word or phrase to get started."
        else:
            try:
                translated_text = translate_text(text, source_language, target_language)
                words = sign_glosses(text, source_language)
                for word in words:
                    sign = Sign.objects.filter(english_word__iexact=word).first()
                    lesson = None
                    if sign and sign.video_embed_url:
                        video_url = sign.video_embed_url
                    else:
                        video_url = ""
                        if sign:
                            category = "greetings" if sign.category == "greeting" else "vocabulary"
                            lesson = Lesson.objects.filter(is_active=True, category=category, video_embed_url__gt="").first()
                            video_url = lesson.video_embed_url if lesson else ""
                    cards.append({"word": word, "sign": sign, "video_url": video_url, "lesson_title": lesson.title if lesson else ""})
                if request.user.is_authenticated:
                    TranslationHistory.objects.create(user=request.user, input_text=text, source_language=source_language, target_language=target_language, translated_text=translated_text)
            except ValueError as exc:
                error = str(exc)
        if source_language not in LANGUAGE_NAMES:
            source_language = "en"
        if target_language not in LANGUAGE_NAMES:
            target_language = "hi"
    return render(request, "translation/translator.html", {
        "text": text, "source_language": source_language, "target_language": target_language,
        "translated_text": translated_text, "cards": cards, "error": error,
        "languages": LANGUAGE_NAMES, "target_language_label": LANGUAGE_NAMES.get(target_language, "Hindi"),
    })
