"""
Basira Arabic Text Preprocessor
================================
NLP preprocessing pipeline for Islamic content verification.
Handles normalization, tokenization, and stopword removal for Arabic text.
"""

import re
import string
import unicodedata

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


_VERSE_DIGITS_TRANS = str.maketrans(
    '٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹',
    '01234567890123456789',
)
_VERSE_INVISIBLE_PATTERN = re.compile(
    r'[\u061C\u200B-\u200F\u202A-\u202E\u2066-\u2069\uFEFF]'
)
_VERSE_QURAN_MARKS_PATTERN = re.compile(r'[\u06D6-\u06ED]')
_VERSE_ALEF_PATTERN = re.compile(r'[آأإٱٲٳٵ]')


def normalize_verse_text(
    text: str,
    *,
    expand_dagger_alef: bool = True,
    alef_maqsura_mode: str = 'keep',
) -> str:
    """Normalize Arabic verse text for exact and partial matching.

    This is intentionally separate from :func:`normalize`: the embedding
    corpus was built with the older normalization, so changing that function
    would invalidate the stored vectors.  Canonical Quran text is never
    modified; callers use this only to compare a user's text with it.

    ``expand_dagger_alef`` controls U+0670 (superscript alef).  Uthmani text
    uses it where modern spelling may either keep a full alef (e.g. تبارك)
    or omit it (e.g. الرحمن), so matching builds both variants.

    ``alef_maqsura_mode`` controls alef maqsura followed by a dagger alef
    (e.g. Uthmani ``أَلْهَىٰكُمُ``).  ``'keep'`` folds it to a plain alef
    maqsura (``الهيكم``); ``'alef'`` expands it to a full alef
    (``الهاكم``), which is how modern simplified Arabic usually spells it.
    """
    if alef_maqsura_mode not in ('keep', 'alef'):
        raise ValueError("alef_maqsura_mode must be 'keep' or 'alef'")

    text = unicodedata.normalize('NFC', text or '')
    text = text.translate(_VERSE_DIGITS_TRANS)
    text = text.replace('\u0640', '')
    text = _VERSE_INVISIBLE_PATTERN.sub('', text)
    text = _VERSE_QURAN_MARKS_PATTERN.sub('', text)
    # Uthmani orthography writes the long ā in words such as الصلوة and
    # الربوا as waw + superscript alef (and sometimes a following plain
    # alef).  The modern spelling uses one plain alef.  Fold the whole
    # sequence to a single alef so a following plain alef does not create a
    # spurious doubled alef.  Do not collapse other doubled alefs: sequences
    # such as "يا" + "إبراهيم" are legitimate word boundaries and collapsing
    # them would make normalization non-idempotent.
    text = text.replace('\u0648\u0670\u0627', 'ا')
    text = text.replace('\u0648\u0670', 'ا')
    if alef_maqsura_mode == 'alef':
        for sequence in ('\u0649\u0670', '\u064a\u0670', '\u06cc\u0670'):
            text = text.replace(sequence, 'ا')
    else:
        for sequence in ('\u0649\u0670', '\u064a\u0670', '\u06cc\u0670'):
            text = text.replace(sequence, sequence[0])
    text = text.replace('\u0670', 'ا' if expand_dagger_alef else '')
    text = text.replace('\u0671', 'ا')
    text = _VERSE_ALEF_PATTERN.sub('ا', text)
    text = text.replace('\u0621', '')
    text = (
        text.replace('ى', 'ي')
        .replace('ی', 'ي')
        .replace('ئ', 'ي')
        .replace('ؤ', 'و')
        .replace('ة', 'ه')
        .replace('ہ', 'ه')
        .replace('ک', 'ك')
    )
    text = ''.join(ch for ch in text if not unicodedata.combining(ch))
    text = ''.join(
        ch if (ch.isalnum() or ch.isspace()) else ' ' for ch in text
    )
    return _EXTRA_SPACES_PATTERN.sub(' ', text).strip()


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
