"""Small offline vocabulary helper; intentionally not a full machine translator."""
import re

LANGUAGE_NAMES = {"en": "English", "hi": "Hindi", "mr": "Marathi"}
WORDS = {
    "hello": {"en": "hello", "hi": "नमस्ते", "mr": "नमस्कार"},
    "hi": {"en": "hi", "hi": "नमस्ते", "mr": "नमस्कार"},
    "thank": {"en": "thank", "hi": "धन्यवाद", "mr": "धन्यवाद"},
    "you": {"en": "you", "hi": "आप", "mr": "तुम्ही"},
    "please": {"en": "please", "hi": "कृपया", "mr": "कृपया"},
    "help": {"en": "help", "hi": "मदद", "mr": "मदत"},
    "water": {"en": "water", "hi": "पानी", "mr": "पाणी"},
    "yes": {"en": "yes", "hi": "हाँ", "mr": "हो"},
    "no": {"en": "no", "hi": "नहीं", "mr": "नाही"},
    "family": {"en": "family", "hi": "परिवार", "mr": "कुटुंब"},
    "book": {"en": "book", "hi": "किताब", "mr": "पुस्तक"},
    "school": {"en": "school", "hi": "विद्यालय", "mr": "शाळा"},
    "learn": {"en": "learn", "hi": "सीखना", "mr": "शिका"},
    "sign": {"en": "sign", "hi": "संकेत", "mr": "खूण"},
    "language": {"en": "language", "hi": "भाषा", "mr": "भाषा"},
    "i": {"en": "I", "hi": "मैं", "mr": "मी"},
    "want": {"en": "want", "hi": "चाहता", "mr": "हवे"},
    "my": {"en": "my", "hi": "मेरा", "mr": "माझे"},
    "name": {"en": "name", "hi": "नाम", "mr": "नाव"},
    "good": {"en": "good", "hi": "अच्छा", "mr": "चांगले"},
    "morning": {"en": "morning", "hi": "सुबह", "mr": "सकाळ"},
}
REVERSE = {language: {value[language].casefold(): key for key, value in WORDS.items()} for language in LANGUAGE_NAMES}
REVERSE["hi"].update({"धन्यवाद": "thank", "चाहती": "want", "चाहता": "want"})
REVERSE["mr"].update({"धन्यवाद": "thank", "हवी": "want"})


def _tokens(text):
    return re.findall(r"[\w]+(?:['’][\w]+)?|[^\w\s]", text, flags=re.UNICODE)


def normalize_to_english(text, source_language):
    reverse = REVERSE[source_language]
    normalized = []
    for token in _tokens(text):
        canonical = reverse.get(token.casefold())
        normalized.append(canonical if canonical else token)
    return normalized


def translate_text(text, source_language="en", target_language="hi"):
    """Translate recognized vocabulary word by word, retaining unknown words."""
    if source_language not in LANGUAGE_NAMES or target_language not in LANGUAGE_NAMES:
        raise ValueError("Choose English, Hindi, or Marathi.")
    canonical = normalize_to_english(text, source_language)
    output = []
    for token in canonical:
        word = WORDS.get(token.casefold())
        output.append(word[target_language] if word else token)
    # Adjacent token joining is language neutral enough for this phrasebook-sized fallback.
    return " ".join(output).replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?")


def sign_glosses(text, source_language="en"):
    """Return recognized English vocabulary and preserve unrecognized words for fingerspelling."""
    canonical = normalize_to_english(text, source_language)
    return [token.casefold() if token.casefold() in WORDS else token for token in canonical if re.search(r"\w", token, flags=re.UNICODE)]
