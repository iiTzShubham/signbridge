from django.contrib import admin

from .models import LearningProgress, Lesson


@admin.register(Lesson)
class LessonAdmin(admin.ModelAdmin):
    list_display = ("title", "category", "duration_minutes", "is_active", "sort_order")
    list_filter = ("category", "is_active")
    search_fields = ("title", "summary", "lesson_text")
    prepopulated_fields = {"slug": ("title",)}
    ordering = ("sort_order", "title")


@admin.register(LearningProgress)
class LearningProgressAdmin(admin.ModelAdmin):
    list_display = ("user", "lesson", "status", "practice_count", "last_viewed_at", "completed_at")
    list_filter = ("status", "lesson__category")
    search_fields = ("user__username", "lesson__title")
    readonly_fields = ("first_viewed_at", "last_viewed_at", "completed_at")
