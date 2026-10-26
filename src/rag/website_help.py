"""
Basira — Website Help
=====================
A small, maintainable source of truth for questions about Basira itself.

The religious RAG pipeline answers questions about Islam from the Quran,
Hadith, Tafsir, Fiqh, and Sira corpus.  Those sources cannot answer questions
such as "how do I search Tafsir?" or "what is the difference between the
knowledge assistant and verification?".  This module supplies those answers
from the implemented product surface only.

It is intentionally data-driven so the website facts stay in one place and
do not drift into the religious evidence path.
"""

from __future__ import annotations

from src.nlp.preprocessor import normalize

WEBSITE_HELP_SOURCE = "موقع بصيرة — دليل الاستخدام"

# ---------------------------------------------------------------------------
# Product facts (keep synchronized with FrontEnd routes/components)
# ---------------------------------------------------------------------------

SECTIONS = {
    "home": {
        "label": "الرئيسية",
        "route": "/",
        "description": "الصفحة التعريفية ببصيرة وخدماتها.",
    },
    "chat": {
        "label": "المساعد المعرفي",
        "route": "/chat",
        "description": (
            "محادثة للأسئلة الشرعية المفتوحة، تجيب من القرآن والحديث "
            "والتفسير والفقه والسيرة مع إظهار المصادر."
        ),
    },
    "verify": {
        "label": "التحقق من المعلومات",
        "route": "/verify",
        "description": (
            "خدمة مستقلة للتحقق من نص أو معلومة، مع إمكانية إضافة تفسير "
            "اختياري لتقييمه. تعرض هل النص موجود واقتباس كامل أم جزئي، "
            "والمصدر، وتقييم التفسير إن قُدّم."
        ),
    },
    "tafsir": {
        "label": "البحث في التفسير",
        "route": "/tafsir",
        "description": (
            "بحث عن آية محددة أو جزء منها لعرض تفسير ابن كثير والسياق "
            "والمصدر."
        ),
    },
    "services": {
        "label": "الخدمات",
        "route": "/services",
        "description": "صفحة تجمع الأدوات الثلاث وتشرح الفرق بينها.",
    },
    "about": {
        "label": "عن بصيرة",
        "route": "/about",
        "description": "صفحة تعريفية عن المشروع وهدفه.",
    },
}

CHAT_FEATURES = (
    "من داخل صفحة المساعد المعرفي، الشريط الجانبي مخصص للمحادثات: "
    "زر «محادثة جديدة»، وخانة بحث في عناوين المحادثات، وقائمة المحادثات "
    "السابقة. لكل محادثة زران لتغيير الاسم أو حذفها. هذه الميزات تظهر "
    "للمستخدم المسجل، لأن المحادثات المحفوظة مرتبطة بحسابه."
)

GUEST_NOTE = (
    "الزائر يستطيع إرسال رسائل أيضًا، لكن المحادثة لا تُحفظ في السجل بعد "
    "إغلاق الصفحة، وحد الرسالة أقل من حد المستخدم المسجل."
)

LIMITS_NOTE = (
    "الحد الأقصى للرسالة 500 حرف للمستخدم المسجل و250 حرفًا للزائر، "
    "وكل محادثة محفوظة لها حد أقصى للرسائل."
)

LOGIN_NOTE = (
    "تسجيل الدخول أو إنشاء حساب من زر الحساب في الشريط العلوي. "
    "بعد الدخول يمكنك فتح «الإعدادات» من صفحة الحساب، والوصول إلى سجل "
    "محادثاتك من الشريط الجانبي في صفحة المساعد."
)

TAFSIR_STEPS = (
    "للبحث عن تفسير آية: افتح «البحث في التفسير» من الشريط العلوي، "
    "ثم اكتب في الخانة اسم السورة ورقم الآية (مثل «النساء:43») أو جزءًا "
    "واضحًا من الآية، واضغط «ابحث عن التفسير». النتيجة تعرض التفسير "
    "والسياق والمصدر عند توفره."
)

VERIFY_STEPS = (
    "لاستخدام التحقق من المعلومات: افتح «التحقق من المعلومات» من الشريط "
    "العلوي، والصق المنشور كاملًا في خانة المنشور، بما فيه الاقتباسات "
    "والشرح والاستنتاج، ثم اضغط «ابدأ التحقق». الأداة تفصل بين صحة النقل، "
    "واكتمال السياق، ودرجة الحديث أو ثبوت الخبر، وصحة شرح المؤلف واستدلاله، "
    "وتعرض النص الكامل والمرجع عند توفره. التحقق خدمة منفصلة عن المساعد، "
    "ولا يعتمد على إجابة المساعد كدليل."
)

CHAT_STEPS = (
    "لبدء محادثة: افتح «المساعد المعرفي» من الشريط العلوي، واكتب سؤالك في "
    "مربع الرسالة ثم اضغط Enter أو زر الإرسال. استخدم Shift+Enter لسطر "
    "جديد. الإجابة تظهر مع المصادر، ويمكنك متابعة السؤال في نفس المحادثة."
)

DIFFERENCE_TEXT = (
    "المساعد المعرفي خدمة حوارية: بتكتب سؤالك الشرعي، وهو يبحث في المصادر "
    "المتاحة (القرآن، الحديث، التفسير، الفقه، السيرة) ويجيب مع ذكر المصادر. "
    "أما التحقق من المعلومات فخدمة مستقلة تأخذ نصًا أو معلومة جاهزة وتفحص "
    "مدى وجودها في المصادر ومدى اكتمالها، ولو أضفت تفسيرًا لها تقيّمه "
    "بشكل منفصل. يعني المساعد يساعدك تفهم وتسأل، والتحقق يساعدك تراجع نصًا "
    "أو تفسيرًا محددًا. الخدمتان منفصلتان في الواجهة والـAPI، ولا تُستخدم "
    "إجابة المساعد كدليل على صحة النص في التحقق."
)

OVERVIEW_TEXT = (
    "بصيرة مساعد معرفي إسلامي: بيساعدك تسأل عن القرآن والحديث والتفسير "
    "والفقه والسيرة، ويجيب من المصادر المتاحة مع إظهار المراجع بدل الاعتماد "
    "على كلام عام. وفيه كمان خدمتان مساعدتان: «التحقق من المعلومات» لفحص "
    "نص أو معلومة ومعرفة مصدرها ومدى اكتمالها، و«البحث في التفسير» للوصول "
    "لتفسير آية محددة. تقدر تبدأ من «المساعد المعرفي» في الشريط العلوي، "
    "وتلاقي وصف الأدوات في صفحة «الخدمات»."
)

THEME_TEXT = (
    "تقدر تغيّر المظهر من زر الشمس/القمر في الشريط العلوي، أو من أسفل "
    "الشريط الجانبي داخل صفحة المساعد. الاختيار محفوظ في المتصفح."
)


# ---------------------------------------------------------------------------
# Lightweight intent detection
# ---------------------------------------------------------------------------

# Normalised because normalize() folds hamza/alef/ta-marbuta and punctuation.
_WORKFLOW_MARKERS = (
    "ازاي", "كيف", "ازاى", "طريقه", "خطوات", "اوصل", "اوصلني", "فين",
    "منين", "وين", "اين", "استخدم", "استخدام", "ابحث", "بحث", "افتح",
    "اضغط", "اختار", "اختار", "ادخل", "سجل", "تسجيل", "حساب", "اعدادات",
    "المظهر", "الوضع الليلي", "الوضع النهاري", "سجل المحادثات",
    "المحادثات القديمه", "محادثه جديده", "جديده", "السابق", "الفرق",
    "ماذا تفعل", "بتعمل", "تساعدني", "بتساعد", "الخدمات", "خدمه",
    "خدمات", "مميزات", "ميزه", "واجهه", "الموقع", "بصيره",
    "الشريط العلوي", "الشريط الجانبي", "القائمه", "اعرف",
    "الحد الاقصي", "طول الرساله", "عدد الرسائل", "الرساله", "رسائل",
)

_SERVICE_MARKERS = (
    "المساعد المعرفي", "التحقق من المعلومات", "خدمه التحقق",
    "البحث في التفسير", "خدمه التفسير", "صفحه التفسير",
    "صفحه التحقق", "صفحه المساعد", "المحادثات",
)

_SITE_ANCHORS = (
    "بصيره", "الموقع", "المنصه", "الواجهه", "الشريط العلوي",
    "الشريط الجانبي", "القائمه", "الخدمات",
)

# Strong religious content.  "تفسير" and "آية" are deliberately excluded:
# they appear in legitimate service questions ("إزاي أبحث عن تفسير آية؟").
_RELIGIOUS_MARKERS = (
    "حكم", "احكام", "يجوز", "حرام", "حلال", "واجب", "فرض", "مكروه",
    "مستحب", "مندوب", "بدعه", "سنه", "سنن", "وضوء", "صلاه", "زكاه",
    "صوم", "صيام", "حج", "عمره", "جنابه", "طهاره", "حديث", "احاديث",
    "سيره", "فقه", "فتوي", "مذهب", "رسول", "نبي", "الله", "القران",
    "معني الايه", "سبب النزول", "تفسير الايه", "فضل", "ثواب", "عقاب",
    "وقف", "اوقاف",
)

_RELIGIOUS_PATTERNS = (
    "ما تفسير", "ما معني", "هل يجوز", "هل يجوز", "ما حكم", "حكم",
    "ما هي", "ما هو", "فضل", "ثواب", "عقاب", "معني قوله",
)

# Words that can belong to either a religious question or a service question
# ("تفسير آية" vs. "إزاي أبحث عن تفسير آية").  They only shift the decision
# when there is no explicit site workflow phrase.
_WEAK_RELIGIOUS_MARKERS = (
    "تفسير", "ايه", "ايات", "سوره", "قران", "معني",
)

_SITE_CONTEXT_MARKERS = (
    "ابحث", "افتح", "استخدم", "اوصل", "فين", "منين", "الشريط",
    "القائمه", "محادثه جديده", "تسجيل", "حساب", "اعدادات", "المظهر",
    "التحقق من المعلومات", "البحث في التفسير", "الصفحه", "صفحه",
    "الموقع", "بصيره", "الخدمات", "تساعدني", "بتساعد", "ماذا تفعل",
)

_SITE_WORKFLOW_MARKERS = (
    "ابحث", "أبحث", "استخدم", "أستخدم", "افتح", "الفرق",
    "المحادثات", "المحادثه", "التحقق من المعلومات",
    "البحث في التفسير", "الصفحه", "صفحه", "الموقع", "بصيره",
)

_TOPIC_MARKERS = {
    "difference": (
        "الفرق بين", "الفرق", "يفرق", "الفرق بين المساعد", "المساعد والتحقق",
    ),
    "tafsir": (
        "التفسير", "تفسير", "tafsir", "ايه الكرسي", "سوره", "الايه",
    ),
    "verify": (
        "التحقق", "تحقق", "verify", "فحص النص", "تفسير اختياري",
    ),
    "history": (
        "المحادثات", "المحادثه", "السجل", "سجل", "القديمه", "جديده",
        "حذف", "تغيير الاسم", "بحث في المحادثات",
    ),
    "account": (
        "تسجيل الدخول", "حساب", "حسابي", "اعدادات", "بروفايل", "كلمه المرور",
    ),
    "theme": (
        "المظهر", "الوضع الليلي", "الوضع النهاري", "الثيم", "الوان",
    ),
    "limits": (
        "الحد", "اقصي", "طول الرساله", "عدد الرسائل", "250", "500",
    ),
    "overview": (
        "ماذا تفعل", "بتعمل", "تساعدني", "بتساعد", "الخدمات", "خدمات",
        "مميزات", "ميزه", "عن الموقع", "ما هو بصيره", "بصيره",
    ),
}


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def classify_website_intent(
    query: str,
    chat_history: list[dict] | None = None,
) -> dict:
    """Classify a question as website help, religious, or mixed.

    The classifier is deliberately conservative:
    - a bare "تفسير آية" question is religious unless it also asks *how* to
      use the Tafsir page;
    - a question with a strong religious subject is never hijacked by a
      generic word such as "خدمة" or "الموقع".
    """
    text = normalize(query or "")
    words = set(text.split())

    workflow_hits = sum(1 for marker in _WORKFLOW_MARKERS if marker in text)
    service_hits = sum(1 for marker in _SERVICE_MARKERS if marker in text)
    anchor = _contains_any(text, _SITE_ANCHORS)
    religious_hits = sum(
        1 for marker in _RELIGIOUS_MARKERS if marker in words or marker in text
    )

    # "ما تفسير آية..." with no usage/navigation phrasing is a religious
    # question, even though it names the Tafsir feature.
    religious_request = _contains_any(text, _RELIGIOUS_PATTERNS)
    usage_request = _contains_any(
        text,
        (
            "ازاي", "كيف", "ازاى", "طريقه", "خطوات", "استخدم", "استخدام",
            "ابحث", "افتح", "اوصل", "فين", "منين", "الفرق", "ماذا تفعل",
            "تساعدني", "بتساعد", "بتعمل",
        ),
    )
    explicit_service = service_hits > 0
    site_specific = (
        anchor
        or explicit_service
        or _contains_any(text, _SITE_CONTEXT_MARKERS)
    )

    # "كيف ينعقد الوقف؟" and "ما الفرق بين الصريح والكناية؟" are religious
    # content questions.  Generic how/compare words must not route them to
    # the website guide unless the question actually names a site feature,
    # page, or workflow.
    if not site_specific and (religious_request or religious_hits > 0):
        return {
            "is_website": False,
            "is_mixed": False,
            "topics": [],
            "score": 0,
            "reason": "religious_content",
        }

    website_score = workflow_hits + service_hits * 2 + (3 if anchor else 0)
    if explicit_service and usage_request:
        website_score += 2
    if "ماذا تفعل" in text or "تساعدني" in text or "بتساعد" in text:
        website_score += 2

    is_website = (
        website_score >= 2
        or anchor
        or (explicit_service and usage_request)
    )

    # A pure website how-to question may contain "تفسير آية" without being a
    # religious request.  Keep it on the website path.
    if usage_request and explicit_service and not religious_hits:
        is_website = True

    weak_religious = _contains_any(text, _WEAK_RELIGIOUS_MARKERS)
    site_workflow = _contains_any(text, _SITE_WORKFLOW_MARKERS)
    if (
        is_website
        and weak_religious
        and not site_workflow
        and not explicit_service
    ):
        # "كيف تساعدني في فهم آية الكرسي؟" is a religious question that
        # happens to use the verb "تساعدني".
        is_website = False

    is_mixed = bool(is_website and religious_hits >= 2)

    topics = _detect_topics(text) if is_website else []
    return {
        "is_website": bool(is_website),
        "is_mixed": is_mixed,
        "topics": topics,
        "score": website_score,
        "religious_score": religious_hits,
        "reason": "website_help" if is_website else "religious",
    }


def _detect_topics(text: str) -> list[str]:
    topics: list[str] = []
    for topic, markers in _TOPIC_MARKERS.items():
        if any(marker in text for marker in markers):
            topics.append(topic)
    # "الفرق" alone is enough only when both service names are present.
    if "difference" in topics and not (
        "المساعد" in text and ("التحقق" in text or "التفسير" in text)
    ):
        topics.remove("difference")
    if not topics:
        topics.append("overview")
    return topics


def build_website_answer(
    query: str,
    chat_history: list[dict] | None = None,
) -> str:
    """Return a natural Arabic answer from the website source of truth."""
    intent = classify_website_intent(query, chat_history)
    topics = intent.get("topics") or ["overview"]
    text = normalize(query or "")

    parts: list[str] = []

    if "difference" in topics:
        parts.append(DIFFERENCE_TEXT)
    elif "tafsir" in topics and any(
        marker in text
        for marker in ("ازاي", "كيف", "ابحث", "اوصل", "استخدم", "فين", "منين")
    ):
        parts.append(TAFSIR_STEPS)
        parts.append(
            "لو الآية مش واضحة، اكتب جزءًا مميزًا منها؛ ولو ظهرت أكثر من "
            "نتيجة، حدد اسم السورة ورقم الآية."
        )
    elif "verify" in topics and any(
        marker in text
        for marker in ("ازاي", "كيف", "استخدم", "افتح", "خطوات", "فين")
    ):
        parts.append(VERIFY_STEPS)
    elif "history" in topics:
        parts.append(CHAT_FEATURES)
        parts.append(GUEST_NOTE)
    elif "account" in topics:
        parts.append(LOGIN_NOTE)
    elif "theme" in topics:
        parts.append(THEME_TEXT)
    elif "limits" in topics:
        parts.append(LIMITS_NOTE)
    elif "overview" in topics or not parts:
        parts.append(OVERVIEW_TEXT)
        if "verify" in topics:
            parts.append(VERIFY_STEPS)
        if "tafsir" in topics:
            parts.append(TAFSIR_STEPS)

    # Add task-specific guidance for broad "how do you help me?" questions.
    if "overview" in topics:
        if any(marker in text for marker in ("كيف تساعدني", "تساعدني", "بتساعد")):
            parts.append(
                "تقدر تسأل سؤالًا شرعيًا مباشرًا، أو تبدأ بوصف مشكلتك، "
                "والمساعد يوضح لك ما تدعمه المصادر. ولو عندك نص جاهز وتريد "
                "مراجعته، استخدم «التحقق من المعلومات» بدل المحادثة."
            )
        if any(marker in text for marker in ("ازاي", "كيف", "استخدم")):
            parts.append(CHAT_STEPS)

    if "chat" not in topics and "history" not in topics:
        parts.append(
            "كل الأدوات موجودة في الشريط العلوي: الرئيسية، المساعد المعرفي، "
            "التحقق من المعلومات، البحث في التفسير، الخدمات، عن بصيرة."
        )

    # De-duplicate while preserving order.
    unique: list[str] = []
    for part in parts:
        if part and part not in unique:
            unique.append(part)
    return "\n\n".join(unique)


def website_help_sources() -> list[str]:
    return [WEBSITE_HELP_SOURCE]
