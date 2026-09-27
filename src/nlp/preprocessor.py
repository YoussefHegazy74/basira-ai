"""
Basira Arabic Text Preprocessor
================================
NLP preprocessing pipeline for Islamic content verification.
Handles normalization, tokenization, and stopword removal for Arabic text.
"""

import re
import string

from pyarabic.araby import strip_tashkeel


# Arabic stopwords list for Islamic/general Arabic text filtering
ARABIC_STOPWORDS = {
    'من', 'في', 'على', 'إلى', 'عن', 'مع',
    'هذا', 'هذه', 'التي', 'الذي',
    'وقد', 'قد', 'كان', 'كانت',
    'أن', 'إن', 'لا', 'ما', 'لم',
    'و', 'أو', 'ثم',
    'هو', 'هي', 'به', 'له',
}

# Precompile regex patterns for performance
_HAMZA_ALEF_PATTERN = re.compile(r'[إأآا]')
_ALEF_WASLA_PATTERN = re.compile(r'\u0671')       # ٱ → ا
_QURANIC_MARKS_PATTERN = re.compile(r'[\u0670\u06D6-\u06ED]')  # Quranic annotation marks + superscript alef
_TA_MARBUTA_PATTERN = re.compile(r'ة')
_PUNCTUATION_PATTERN = re.compile(
    r'[' + re.escape(string.punctuation) + r'؟،؛»«﴿﴾۞٪' + r']'
)
_EXTRA_SPACES_PATTERN = re.compile(r'\s+')


def normalize(text: str) -> str:
    """Normalize Arabic text for downstream NLP processing.

    Steps applied in order:
    1. Remove tashkeel (diacritical marks) via pyarabic
    2. Replace alef wasla (ٱ U+0671) → ا
    3. Remove Quranic annotation marks (U+06D6–U+06ED)
    4. Unify hamza forms (إ أ آ ا) → ا
    5. Unify ta marbuta (ة) → ه
    6. Remove punctuation (Latin + Arabic)
    7. Collapse extra whitespace and strip
    """
    text = strip_tashkeel(text)
    text = _ALEF_WASLA_PATTERN.sub('ا', text)
    text = _QURANIC_MARKS_PATTERN.sub('', text)
    text = _HAMZA_ALEF_PATTERN.sub('ا', text)
    text = _TA_MARBUTA_PATTERN.sub('ه', text)
    text = _PUNCTUATION_PATTERN.sub(' ', text)
    text = _EXTRA_SPACES_PATTERN.sub(' ', text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Split normalized Arabic text into word tokens.

    Applies normalization first, then splits on whitespace.
    Returns a list of non-empty token strings.
    """
    normalized = normalize(text)
    return normalized.split()


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Remove Arabic stopwords from a token list.

    Uses the predefined ARABIC_STOPWORDS set. Also filters out
    normalized variants (stopwords are checked after applying
    the same hamza/ta-marbuta unification).
    """
    # Normalize the stopwords the same way we normalize text so matching works
    normalized_stops = set()
    for sw in ARABIC_STOPWORDS:
        nsw = _HAMZA_ALEF_PATTERN.sub('ا', sw)
        nsw = _TA_MARBUTA_PATTERN.sub('ه', nsw)
        normalized_stops.add(nsw)

    return [t for t in tokens if t not in normalized_stops]


def preprocess(text: str) -> dict:
    """Run the full preprocessing pipeline on Arabic text.

    Pipeline: normalize → tokenize → remove_stopwords

    Returns:
        dict with keys:
            - original:     the raw input text
            - normalized:   text after normalization
            - tokens:       list of tokens (post-normalization)
            - tokens_clean: list of tokens after stopword removal
    """
    normalized = normalize(text)
    tokens = normalized.split()
    tokens_clean = remove_stopwords(tokens)

    return {
        'original': text,
        'normalized': normalized,
        'tokens': tokens,
        'tokens_clean': tokens_clean,
    }


if __name__ == '__main__':
    ayah = 'بِسۡمِ ٱللَّهِ ٱلرَّحۡمَٰنِ ٱلرَّحِيمِ'
    result = preprocess(ayah)

    print('=== Basira Preprocessor Test ===')
    for key, value in result.items():
        print(f'{key}: {value}')
