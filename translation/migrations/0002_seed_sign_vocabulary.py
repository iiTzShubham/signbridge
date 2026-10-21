from django.db import migrations


SIGNS = [
    ("hello", "नमस्ते", "नमस्कार", "A greeting used to begin a conversation.", "greeting", "Try the greeting lesson and notice the signer’s expression."),
    ("hi", "नमस्ते", "नमस्कार", "A friendly greeting.", "greeting", "Use this as a simple greeting prompt."),
    ("thank", "धन्यवाद", "धन्यवाद", "A polite word for showing appreciation.", "daily", "Practice the complete phrase “thank you” in context."),
    ("you", "आप", "तुम्ही", "A word for the person you are speaking with.", "people", "Look at the person while communicating."),
    ("please", "कृपया", "कृपया", "A polite request prompt.", "daily", "Use facial expression and context in a polite request."),
    ("help", "मदद", "मदत", "A useful word to ask someone for assistance.", "daily", "Practice this word inside a short request."),
    ("water", "पानी", "पाणी", "A common everyday word.", "daily", "Try this word in a simple everyday conversation."),
    ("yes", "हाँ", "हो", "An affirmative response.", "daily", "Practice a natural affirmative response."),
    ("no", "नहीं", "नाही", "A negative response.", "daily", "Practice a natural negative response."),
    ("family", "परिवार", "कुटुंब", "A word for family and relatives.", "people", "Think of the people in your family as you practise."),
    ("book", "किताब", "पुस्तक", "A familiar object and learning word.", "learning", "Try using this word while talking about study."),
    ("school", "विद्यालय", "शाळा", "A place for learning.", "learning", "Try this word in a sentence about learning."),
    ("learn", "सीखना", "शिका", "A word for learning a new skill.", "learning", "Repeat after the signer in the learning lesson."),
    ("sign", "संकेत", "खूण", "A prompt related to signed communication.", "learning", "Follow the ISL lesson rather than translating word by word."),
    ("language", "भाषा", "भाषा", "A word for a language or way of communicating.", "learning", "ISL has its own grammar and visual language structure."),
    ("i", "मैं", "मी", "A word used when speaking about yourself.", "people", "Use eye gaze and facial expression with the full phrase."),
    ("want", "चाहता", "हवे", "A word expressing a want or need.", "daily", "The best ISL expression depends on the full context."),
    ("my", "मेरा", "माझे", "A word showing something belongs to you.", "people", "Use this as a vocabulary prompt, not a word-for-word rule."),
    ("name", "नाम", "नाव", "A word used when introducing yourself.", "greeting", "The ISLRTC introduction lesson shows names in context."),
    ("good", "अच्छा", "चांगले", "A common descriptive word.", "daily", "Context and facial expression can change how it is used."),
    ("morning", "सुबह", "सकाळ", "A time-of-day word used in greetings.", "greeting", "Try it with the ISLRTC greeting video."),
]


def add_vocabulary(apps, schema_editor):
    Sign = apps.get_model("translation", "Sign")
    Lesson = apps.get_model("learning", "Lesson")
    lesson_by_category = {
        "greeting": Lesson.objects.filter(slug="greetings-in-isl").first(),
        "daily": Lesson.objects.filter(slug="common-everyday-words-in-isl").first(),
        "people": Lesson.objects.filter(slug="introduce-yourself-in-isl").first(),
        "learning": Lesson.objects.filter(slug="common-everyday-words-in-isl").first(),
    }
    for index, (english, hindi, marathi, description, category, example) in enumerate(SIGNS, start=1):
        lesson = lesson_by_category.get(category)
        Sign.objects.update_or_create(english_word=english, defaults={
            "hindi_word": hindi, "marathi_word": marathi, "description": description,
            "category": category, "difficulty": "Beginner", "example_sentence": example,
            "video_embed_url": lesson.video_embed_url if lesson else "",
            "source_url": lesson.source_url if lesson else "", "sort_order": index,
        })


def remove_vocabulary(apps, schema_editor):
    apps.get_model("translation", "Sign").objects.filter(english_word__in=[row[0] for row in SIGNS]).delete()


class Migration(migrations.Migration):
    dependencies = [
        ("learning", "0002_lesson_image_path"),
        ("translation", "0001_initial"),
    ]

    operations = [migrations.RunPython(add_vocabulary, remove_vocabulary)]
