from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db.models import Q
from django.http import HttpResponseNotAllowed
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone

from .models import LearningProgress, Lesson


def lesson_list(request):
    lessons = Lesson.objects.filter(is_active=True)
    category = request.GET.get("category", "")
    query = request.GET.get("q", "").strip()
    if category and category in Lesson.Category.values:
        lessons = lessons.filter(category=category)
    if query:
        lessons = lessons.filter(Q(title__icontains=query) | Q(summary__icontains=query) | Q(lesson_text__icontains=query))

    completed_ids = set()
    completed_count = 0
    if request.user.is_authenticated:
        user_progress = LearningProgress.objects.filter(user=request.user)
        completed_ids = set(user_progress.filter(status=LearningProgress.Status.COMPLETED).values_list("lesson_id", flat=True))
        completed_count = len(completed_ids)
    all_lessons_count = Lesson.objects.filter(is_active=True).count()
    categories = list(Lesson.objects.filter(is_active=True).values_list("category", flat=True).distinct())
    return render(request, "learning/lesson_list.html", {
        "lessons": lessons,
        "active_category": category,
        "search_query": query,
        "categories": categories,
        "completed_ids": completed_ids,
        "completed_count": completed_count,
        "all_lessons_count": all_lessons_count,
    })


def lesson_detail(request, slug):
    lesson = get_object_or_404(Lesson, slug=slug, is_active=True)
    progress = None
    if request.user.is_authenticated:
        progress, _ = LearningProgress.objects.get_or_create(user=request.user, lesson=lesson)
        if progress.status != LearningProgress.Status.COMPLETED:
            progress.save(update_fields=("last_viewed_at",))
    return render(request, "learning/lesson_detail.html", {"lesson": lesson, "progress": progress})


@login_required
def practice(request, slug):
    if request.method != "POST":
        return HttpResponseNotAllowed(("POST",))
    lesson = get_object_or_404(Lesson, slug=slug, is_active=True)
    progress, _ = LearningProgress.objects.get_or_create(user=request.user, lesson=lesson)
    progress.practice_count += 1
    progress.save(update_fields=("practice_count", "last_viewed_at"))
    messages.success(request, f"Practice session saved for {lesson.title}.")
    return redirect("learning:lesson_detail", slug=slug)


@login_required
def complete(request, slug):
    if request.method != "POST":
        return HttpResponseNotAllowed(("POST",))
    lesson = get_object_or_404(Lesson, slug=slug, is_active=True)
    progress, _ = LearningProgress.objects.get_or_create(user=request.user, lesson=lesson)
    progress.status = LearningProgress.Status.COMPLETED
    progress.completed_at = progress.completed_at or timezone.now()
    progress.save(update_fields=("status", "completed_at", "last_viewed_at"))
    messages.success(request, f"Nice work — {lesson.title} is complete.")
    return redirect("learning:lesson_detail", slug=slug)
