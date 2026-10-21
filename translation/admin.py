from django.contrib import admin
from .models import Sign, TranslationHistory


@admin.register(Sign)
class SignAdmin(admin.ModelAdmin):
    list_display = ("english_word", "hindi_word", "marathi_word", "category", "difficulty")
    list_filter = ("category", "difficulty")
    search_fields = ("english_word", "hindi_word", "marathi_word", "description")


@admin.register(TranslationHistory)
class TranslationHistoryAdmin(admin.ModelAdmin):
    list_display = ("input_text", "source_language", "target_language", "user", "created_at")
    list_filter = ("source_language", "target_language", "created_at")
