"""Annotated offline evaluation set for the Basira RAG pipeline.

Every evidence chunk ID below was verified by reading the actual chunk text
in data/processed/chunks.pkl (normalized), not by keyword guessing.

EVAL_QUESTIONS fields:
    id              stable identifier
    category        evaluation category (see CATEGORIES below)
    question        user query (Arabic)
    answerable      True if the corpus contains sufficient evidence
    evidence_ids    chunk IDs whose text supports the expected answer
                    (any one of them is sufficient unless noted)
    required_facts  facts the answer must state to be correct
    answer_criteria short human-readable acceptance criterion for answers
    sem_proxy_id    OPTIONAL: corpus chunk whose REAL stored embedding is
                    used as a stand-in for the API-encoded query vector when
                    running the harness in --sem-vectors mode.  This mocks
                    ONLY the query-encoding step (OpenAI API is unavailable
                    offline); FAISS search, fusion, selection, gating and
                    grounding all run as genuine production code.  Default:
                    first evidence_id for answerable questions.
    notes           annotation provenance / caveats

GROUNDING_CASES fields:
    id          stable identifier
    answer      LLM-style sentence(s) (with citation markers)
    chunk_ids   retrieved sources the sentence may cite (ordered, 1-based)
    expect      "keep" or "drop"
    absent      OPTIONAL list of substrings that must NOT appear in the
                grounding output even if the overall result is kept
                (sentence-level citation/claim scoping tests)
    why         the capability the case exercises

CATEGORIES (mission Phase C behaviours):
    direct_factual, paraphrase, quran_tafsir, hadith, hadith_primary,
    fiqh_ruling, date_number_entity, multi_chunk, multi_part, ambiguous,
    unanswerable, terminology_trap
"""

# ---------------------------------------------------------------------------
# Answerable + unanswerable questions
# ---------------------------------------------------------------------------

EVAL_QUESTIONS = [
    {
        "id": "badr_date",
        "category": "date_number_entity",
        "question": "متى غزوة بدر؟",
        "answerable": True,
        "evidence_ids": ["sira_646", "fiqh_1184", "tafsir_ibnkathir_8_41"],
        "required_facts": [
            "يوم الجمعة",
            "سبع عشرة من شهر رمضان",
            "(زيادة: السنة الثانية من الهجرة)",
        ],
        "answer_criteria": "يذكر السابع عشر من رمضان (أو 17 رمضان) على الأقل",
        "notes": (
            "sira_646 normalized: 'فكانت وقعه بدر يوم الجمعه صبيحه سبع عشره من شهر رمضان'. "
            "fiqh_1184: 'يوم السابع عشر من رمضان من السنه الثانيه من الهجره في بدر'. "
            "tafsir 8:41: 'لتسع عشره او سبع عشره مضت من رمضان'. "
            "Chunk contains وقعه not غزوه -> only synonym bridging recalls it."
        ),
    },
    {
        "id": "badr_date_paraphrase",
        "category": "paraphrase",
        "question": "في أي يوم وقعت معركة بدر؟",
        "answerable": True,
        "evidence_ids": ["sira_646", "fiqh_1184", "tafsir_ibnkathir_8_41"],
        "required_facts": ["يوم الجمعة", "سبع عشرة من رمضان"],
        "answer_criteria": "يذكر السابع عشر من رمضان",
        "notes": "Same evidence as badr_date; query uses معركة instead of غزوة.",
    },
    {
        "id": "rawatib_confirmed",
        "category": "terminology_trap",
        "question": "ما هي السنن المؤكدة في الصلاة؟",
        "answerable": True,
        "evidence_ids": ["fiqh_160", "fiqh_164", "fiqh_165"],
        "required_facts": [
            "السنن الراتبة تشمل سنة الفجر والظهر والعصر والمغرب والعشاء",
        ],
        "answer_criteria": (
            "يعدد صلوات السنن الراتبة الخمس (فجر، ظهر، عصر، مغرب، عشاء). "
            "fiqh_160 is the decisive chunk; fiqh_164/165 acceptable as "
            "secondary (التطوع chapters)."
        ),
        "sem_proxy_id": "fiqh_160",
        "notes": (
            "fiqh_160: 'ويسمي السنن الراتبة، ويشمل سنة الفجر والظهر والعصر والمغرب والعشاء'. "
            "The phrase المؤكدة is used in the corpus for multiple stressed sunnahs "
            "(الوتر fiqh_165، الجماعة fiqh_195، الكسوف fiqh_184، العقيقة fiqh_1477); fiqh_164 "
            "explicitly contrasts 'السنن غير المؤكدة' with 'السنن والرواتب' inside the "
            "prayer chapter -> in the prayer context the mapping to الراتبة is "
            "corpus-grounded, elsewhere it is not. This question now qualifies في الصلاة; "
            "the unqualified variant is sunan_ambiguous below."
        ),
    },
    {
        "id": "rawatib_paraphrase",
        "category": "paraphrase",
        "question": "ما هي الصلوات المسنونة الرواتب؟",
        "answerable": True,
        "evidence_ids": ["fiqh_160", "fiqh_164"],
        "required_facts": ["سنة الفجر والظهر والعصر والمغرب والعشاء"],
        "answer_criteria": "يعدد صلوات السنن الراتبة الخمس",
        "sem_proxy_id": "fiqh_160",
        "notes": (
            "Query uses رواتب (plural) while fiqh_160 uses الراتبة; رواتب occurs in the "
            "corpus mainly as والرواتب (prefix form) -> tokenization bridge required."
        ),
    },
    {
        "id": "witr_ruling",
        "category": "fiqh_ruling",
        "question": "ما حكم صلاة الوتر؟",
        "answerable": True,
        "evidence_ids": ["fiqh_165"],
        "required_facts": ["الوتر سنة مؤكدة", "حث عليه النبي"],
        "answer_criteria": "يقرر أن الوتر سنة مؤكدة (لا واجب)",
        "sem_proxy_id": "fiqh_165",
        "notes": "fiqh_165: 'الوتر سنة مؤكدة حث عليه الرسول صلى الله عليه وسلم ورغب فيه'. "
                 "Terminology trap: العقيقة fiqh_1477 shares 'سنة مؤكدة' wording.",
    },
    {
        "id": "aqiqa_ruling",
        "category": "terminology_trap",
        "question": "ما حكم العقيقة؟",
        "answerable": True,
        "evidence_ids": ["fiqh_1477", "fiqh_1476"],
        "required_facts": ["العقيقة سنة مؤكدة"],
        "answer_criteria": "يقرر أنها سنة مؤكدة تُذبح عن المولود",
        "sem_proxy_id": "fiqh_1477",
        "notes": (
            "fiqh_1477: 'والعقيقة سنة مؤكدة ولو كان الاب معسرا'. Positive control: the "
            "chunk that traps the rawatib query must still answer its own topic."
        ),
    },
    {
        "id": "orphan_wealth",
        "category": "quran_tafsir",
        "question": "ما جزاء الذين يأكلون أموال اليتامى؟",
        "answerable": True,
        "evidence_ids": ["quran_4_10"],
        "required_facts": ["يأكلون في بطونهم نارا", "سيصلون سعيرا"],
        "answer_criteria": "يذكر وعيد النار / أكل النار في البطون",
        "sem_proxy_id": "quran_4_10",
        "notes": (
            "quran_4_10 (النساء:10). NOTE: quran clean_text drops the superscript alef "
            "(أموال -> امول, يتامى -> يتمى) while user wording keeps it -> lexical gap."
        ),
    },
    {
        "id": "jannah_doors",
        "category": "direct_factual",
        "question": "كم عدد أبواب الجنة؟",
        "answerable": True,
        "evidence_ids": ["hadith_darimi_39619"],
        "required_facts": ["ثمانية أبواب"],
        "answer_criteria": "يذكر الرقم ثمانية",
        "sem_proxy_id": "hadith_darimi_39619",
        "notes": "hadith_darimi_39619 full matn: 'لِلْجَنَّةِ ثَمَانِيَةُ أَبْوَابٍ'.",
    },
    {
        "id": "khandaq_year",
        "category": "date_number_entity",
        "question": "في أي سنة كانت غزوة الخندق؟",
        "answerable": True,
        "evidence_ids": ["sira_952", "sira_942", "sira_930"],
        "required_facts": ["في شوال سنة خمس"],
        "answer_criteria": "يذكر السنة الخامسة للهجرة",
        "sem_proxy_id": "sira_952",
        "notes": (
            "sira_952 original text contains 'غزوة بني قريظة في سنة خمس' and the chunk "
            "reference is 'غزوة الخندق في شوال سنة خمس (21/21)'. The year also appears "
            "in neighboring chunk references of the same chapter."
        ),
    },
    {
        "id": "jamah_ruling",
        "category": "fiqh_ruling",
        "question": "ما حكم صلاة الجماعة؟",
        "answerable": True,
        "evidence_ids": ["fiqh_195"],
        "required_facts": [
            "صلاة الجماعة سنة مؤكدة",
            "أفضل من صلاة الفذ بسبع وعشرين درجة",
        ],
        "answer_criteria": "يقرر أنها سنة مؤكدة ويذكر فضلها",
        "sem_proxy_id": "fiqh_195",
        "notes": (
            "fiqh_195: 'صلاه الجماعه سنه مؤكده ٢ ورد في فضلها احاديث كثيره... صلاه الجماعه "
            "افضل من صلاه الفذ بسبع وعشرين درجه'. Another 'سنة مؤكدة' colliding topic."
        ),
    },
    {
        "id": "ghusl_jumuah",
        "category": "fiqh_ruling",
        "question": "ما حكم الغسل يوم الجمعة؟",
        "answerable": True,
        "evidence_ids": ["fiqh_58", "fiqh_59"],
        "required_facts": ["من الأغسال المستحبة", "أمر الشارع به وأكده", "وقته بالفراغ من الصلاة"],
        "answer_criteria": "يقرر استحبابه ويبين وقته",
        "sem_proxy_id": "fiqh_58",
        "notes": (
            "fiqh_58: 'غسل الجمعه لما كان يوم الجمعه يوم اجتماع للعباده والصلاه امر الشارع "
            "بالغسل واكده...' fiqh_59: 'ويخرج وقت الغسل بالفراغ من الصلاه'. Two chunks, "
            "same source type (fiqh)."
        ),
    },
    {
        "id": "arafah_fasting",
        "category": "multi_chunk",
        "question": "ما حكم صيام يوم عرفة؟",
        "answerable": True,
        "evidence_ids": ["hadith_abudawud_22930", "fiqh_612"],
        "required_facts": [
            "يكفر السنة التي قبله والسنة التي بعده (لغير الحاج)",
            "نهى عن صومه للحاج بعرفة",
        ],
        "answer_criteria": "يجمع بين الفضل (التكفير) والاستثناء (الحاج)",
        "sem_proxy_id": "fiqh_612",
        "notes": (
            "hadith_abudawud_22930 (سنن أبي داود 2426, حديث أبي قتادة): 'وصيام عرفه اني "
            "احتسب على الله ان يكفر السنه التي قبله والسنه التي بعده'. fiqh_612: 'ثبت "
            "عنه انه نهى عن صوم يوم عرفه بعرفات'. Needs evidence from TWO sources "
            "(hadith + fiqh) for a complete answer."
        ),
    },
    {
        "id": "kahf_story",
        "category": "quran_tafsir",
        "question": "ما قصة أصحاب الكهف؟",
        "answerable": True,
        "evidence_ids": ["quran_18_9", "tafsir_ibnkathir_18_10", "quran_18_13"],
        "required_facts": ["أصحاب الكهف والرقيم", "فتية آمنوا بربهم"],
        "answer_criteria": "يشير إلى القصة ومصدرها القرآني/التفسيري",
        "sem_proxy_id": "tafsir_ibnkathir_18_10",
        "notes": (
            "quran_18_9: 'أم حسبت أن أصحاب الكهف والرقيم كانوا من آياتنا عجبا'. "
            "tafsir_ibnkathir_18_10: 'هذا اخبار عن قصه اصحاب... اهل الكهف والرقيم'. "
            "quran_18_13: 'إنهم فتية آمنوا بربهم وزدناهم هدى' supplies the second "
            "required fact; the two earlier evidence chunks do not contain it. "
            "Query token أصحاب matches tafsir/sira spelling; quran chunk uses the "
            "dagger-alef rasm ('اصحب') -> lexical gap covered by كهف/الرقيم tokens."
        ),
    },
    {
        "id": "qadr_night",
        "category": "quran_tafsir",
        "question": "ما هي ليلة القدر؟",
        "answerable": True,
        "evidence_ids": ["quran_97_1", "tafsir_ibnkathir_97_1"],
        "required_facts": ["الليلة المباركة التي أنزل فيها القرآن", "من شهر رمضان"],
        "answer_criteria": "يعرّفها بأنها الليلة التي أنزل فيها القرآن في رمضان",
        "sem_proxy_id": "tafsir_ibnkathir_97_1",
        "notes": (
            "quran_97_1: 'إنا أنزلناه في ليلة القدر'. tafsir_ibnkathir_97_1: 'الليله "
            "المباركه التي قال الله عز وجل انا انزلناه في ليله مباركه... وهي من شهر رمضان'."
        ),
    },
    {
        "id": "rawatib_witr_multi",
        "category": "multi_part",
        "question": "ما هي السنن المؤكدة في الصلاة وما حكم صلاة الوتر؟",
        "answerable": True,
        "evidence_ids": ["fiqh_160", "fiqh_165"],
        "required_facts": [
            "السنن الراتبة: الفجر والظهر والعصر والمغرب والعشاء",
            "الوتر سنة مؤكدة",
        ],
        "answer_criteria": "يجيب عن الشقين: عدد الرواتب + حكم الوتر",
        "sem_proxy_id": "fiqh_160",
        "notes": (
            "Multi-part question requiring TWO chunks from the SAME source type "
            "(fiqh_160 rawatib list + fiqh_165 witr ruling). Stress-tests the "
            "per-type cap in diversity selection."
        ),
    },
    {
        "id": "aqiqa_multi",
        "category": "multi_part",
        "question": "ما حكم العقيقة ومتى تذبح؟",
        "answerable": True,
        "evidence_ids": ["fiqh_1477", "fiqh_1480"],
        "required_facts": [
            "سنة مؤكدة",
            "يوم السابع وإلا فالرابع عشر وإلا الحادي والعشرون",
        ],
        "answer_criteria": "يجيب عن الحكم والوقت معاً",
        "sem_proxy_id": "fiqh_1477",
        "notes": (
            "fiqh_1477 (حكمها) + fiqh_1480 (وقت الذبح: 'لسبع ولاربع عشر ولاحدي وعشرين'). "
            "Two chunks, same source type."
        ),
    },
    {
        "id": "sunan_ambiguous",
        "category": "ambiguous",
        "question": "ما هي السنن المؤكدة؟",
        "answerable": True,
        "evidence_ids": [
            "fiqh_160", "fiqh_164", "fiqh_165", "fiqh_195", "fiqh_184", "fiqh_1477",
        ],
        "required_facts": [],
        "answer_criteria": (
            "إجابة مؤهلة: إما طلب توضيح، أو ذكر أن المصطلح يشمل أكثر من باب "
            "(رواتب الصلاة، الوتر، الجماعة، الكسوف، العقيقة...). يقاس أمن إجابة "
            "مدعومة بأي من الأدلة المقبولة."
        ),
        "sem_proxy_id": "fiqh_165",
        "notes": (
            "Corpus inspection shows سنة مؤكدة is used for: رواتب الصلاة (fiqh_160/164), "
            "الوتر (fiqh_165), الجماعة (fiqh_195), الكسوف (fiqh_184), العقيقة (fiqh_1477). "
            "There is NO single correct list; a good system qualifies or enumerates. "
            "Retrieval acceptance = at least one acceptable evidence chunk in context."
        ),
    },
    {
        "id": "ghush_buyer_option",
        "category": "hadith_primary",
        "question": "ما حكم الغش في البيع وما خيار المشتري عند الغش؟",
        "answerable": True,
        "evidence_ids": ["hadith_tirmidhi_27106", "fiqh_1310"],
        "required_facts": [
            "الغش محرم (حرمة التدليس على المشتري)",
            "من غش فليس منا",
            "للمشتري خيار الرد عند اكتشاف الغش",
        ],
        "answer_criteria": (
            "يذكر حرمة الغش في البيع مع دليله من السنة، ويذكر أن للمشتري "
            "خيار الرد عند التدليس."
        ),
        "sem_proxy_id": "hadith_tirmidhi_27106",
        "notes": (
            "First hadith-primary question: evidence spans a hadith collection "
            "(أبي هريرة، قصة صبرة الطعام: 'أفلا جعلته فوق الطعام حتى يراه "
            "الناس' + 'غش فليس منا') and a fiqh chapter (fiqh_1310: "
            "'وللمشتري خيار الرد ثلاثه ايام... اما الحرمه فللغش والتغرير'). "
            "The topic is deliberately phrased with خيار المشتري so the fiqh "
            "chunk is lexically decisive while the hadith chunk is "
            "semantically decisive (verified: 2/2 in lexical and proxy modes). "
            "Caveat: fiqh_1310 also quotes a shortened hadith variant "
            "('غشنا فليس منا'), so it alone answers the ruling part; the full "
            "wording and the food-pile story live only in hadith_tirmidhi_27106."
        ),
    },
    {
        "id": "unanswerable_country",
        "category": "unanswerable",
        "question": "ما هي عاصمة أستراليا؟",
        "answerable": False,
        "evidence_ids": [],
        "required_facts": [],
        "answer_criteria": "رفض: لا توجد أدلة في المصادر",
        "notes": "No chunk in the corpus covers modern world geography.",
    },
    {
        "id": "unanswerable_worldcup",
        "category": "unanswerable",
        "question": "من فاز بكأس العالم 2022؟",
        "answerable": False,
        "evidence_ids": [],
        "required_facts": [],
        "answer_criteria": "رفض: لا توجد أدلة في المصادر",
        "notes": "Modern sports fact, absent from all five sources.",
    },
    {
        "id": "unanswerable_canada",
        "category": "unanswerable",
        "question": "ما هي عاصمة كندا؟",
        "answerable": False,
        "evidence_ids": [],
        "required_facts": [],
        "answer_criteria": "رفض: لا توجد أدلة في المصادر",
        "notes": "Modern world geography, absent from the corpus.",
    },
    {
        "id": "unanswerable_credit_card",
        "category": "unanswerable",
        "question": "ما حكم استخدام بطاقات الائتمان في الشراء عبر الإنترنت؟",
        "answerable": False,
        "evidence_ids": [],
        "required_facts": [],
        "answer_criteria": "رفض: لا توجد أدلة في المصادر",
        "notes": (
            "Religious-commercial topic (fiqh muamalat) that is nonetheless "
            "absent: df checks show ائتمان=0, بطاقات=0, انترنت=0 in the whole "
            "corpus; the best lexical hit is an unrelated hadith about شراء "
            "بطون الأنعام with coverage 1/6 < gate minimum.  Unlike the other "
            "unanswerables this question LOOKS in-domain, so it probes the "
            "gate's precision on plausible-looking queries."
        ),
    },
]

# ---------------------------------------------------------------------------
# Grounding acceptance/rejection cases (executed directly against _ground_answer)
# ---------------------------------------------------------------------------

GROUNDING_CASES = [
    {
        "id": "g_badr_standard",
        "answer": "كانت غزوة بدر يوم الجمعة في السابع عشر من شهر رمضان (مصدر 1).",
        "chunk_ids": ["sira_646"],
        "expect": "keep",
        "why": "correct paraphrase with ordinal date vs spelled-out source",
    },
    {
        "id": "g_badr_meraka",
        "answer": "وقعت معركة بدر يوم الجمعة في السابع عشر من رمضان (مصدر 1).",
        "chunk_ids": ["sira_646"],
        "expect": "keep",
        "why": "third member of the battle-wording family (معركة) must match وقعة",
    },
    {
        "id": "g_badr_year",
        "answer": "وقعت غزوة بدر في السابع عشر من رمضان من السنة الثانية للهجرة (مصدر 1).",
        "chunk_ids": ["fiqh_1184"],
        "expect": "keep",
        "why": "correct paraphrase including hijri year",
    },
    {
        "id": "g_rawatib_verbatim",
        "answer": "السنن الراتبة هي سنة الفجر والظهر والعصر والمغرب والعشاء (مصدر 1).",
        "chunk_ids": ["fiqh_160"],
        "expect": "keep",
        "why": "near-verbatim control that must stay accepted",
    },
    {
        "id": "g_rawatib_paraphrase",
        "answer": "السنن المؤكدة تشمل صلاة سنة الفجر وسنة الظهر والعصر والمغرب والعشاء (مصدر 1).",
        "chunk_ids": ["fiqh_160"],
        "expect": "keep",
        "why": "paraphrase using المؤكدة terminology for the rawatib list (scoped alias)",
    },
    {
        "id": "g_witr",
        "answer": "صلاة الوتر سنة مؤكدة حث عليها النبي صلى الله عليه وسلم (مصدر 1).",
        "chunk_ids": ["fiqh_165"],
        "expect": "keep",
        "why": "correct fiqh paraphrase",
    },
    {
        "id": "g_jamah",
        "answer": "صلاة الجماعة سنة مؤكدة وأفضل من صلاة الفذ بسبع وعشرين درجة (مصدر 1).",
        "chunk_ids": ["fiqh_195"],
        "expect": "keep",
        "why": "different topic using the same qualifier مؤكدة (no cross-topic contamination)",
    },
    {
        "id": "g_jannah",
        "answer": "للجنة ثمانية أبواب (مصدر 1).",
        "chunk_ids": ["hadith_darimi_39619"],
        "expect": "keep",
        "why": "short numeric claim",
    },
    {
        "id": "g_orphans",
        "answer": "الذين يأكلون أموال اليتامى ظلما إنما يأكلون في بطونهم نارا (مصدر 1).",
        "chunk_ids": ["quran_4_10"],
        "expect": "keep",
        "why": "quran paraphrase; source dropped superscript alef (امول/يتمى)",
    },
    {
        "id": "g_badr_ascii_digits",
        "answer": "وقعت غزوة بدر في 17 من شهر رمضان (مصدر 1).",
        "chunk_ids": ["sira_646"],
        "expect": "keep",
        "why": "digit form of the date (ASCII) vs spelled-out source",
    },
    {
        "id": "g_badr_arabic_indic",
        "answer": "ووقعت غزوة بدر في ١٧ من شهر رمضان (مصدر 1).",
        "chunk_ids": ["sira_646"],
        "expect": "keep",
        "why": "Arabic-Indic digits variant",
    },
    {
        "id": "g_two_citations",
        "answer": "الوتر سنة مؤكدة (مصدر 1). وصلاة الجماعة سنة مؤكدة (مصدر 2).",
        "chunk_ids": ["fiqh_165", "fiqh_195"],
        "expect": "keep",
        "why": "two sentences each supported by its own cited chunk",
    },
    {
        "id": "g_arafah",
        "answer": "صيام يوم عرفة يكفر السنة التي قبله والسنة التي بعده (مصدر 1).",
        "chunk_ids": ["hadith_abudawud_22930"],
        "expect": "keep",
        "why": "hadith wording paraphrase (the fiqh exception for the pilgrim lives in the retrieval question)",
    },
    {
        "id": "g_negation_mirror",
        "answer": "من اغتسل بعد الصلاة لا يعيد الغسل (مصدر 1).",
        "chunk_ids": ["fiqh_59"],
        "expect": "keep",
        "why": (
            "claim preserves the source negation (ولا يعيد الغسل): the "
            "negation-flip check must never reject same-polarity claims"
        ),
    },
    {
        "id": "g_ghusl_negation_elsewhere",
        "answer": "فمن اغتسل بعد الصلاة يكون غسله للجمعة (مصدر 1).",
        "chunk_ids": ["fiqh_59"],
        "expect": "keep",
        "why": (
            "a negation elsewhere in the chunk (ولا يعيد, ولا يعتبر) must not "
            "block affirmative support for the claim's other tokens"
        ),
    },
    {
        "id": "d_badr_wrong_month",
        "answer": "وقعت غزوة بدر يوم الجمعة في شهر شوال (مصدر 1).",
        "chunk_ids": ["sira_646"],
        "expect": "drop",
        "why": (
            "wrong month only: everything else matches the source, so coverage "
            "alone would accept it; the critical month token شوال must reject it"
        ),
    },
    {
        "id": "d_badr_wrong_day",
        "answer": "كانت غزوة بدر في تسع عشرة من رمضان (مصدر 1).",
        "chunk_ids": ["sira_646"],
        "expect": "drop",
        "why": "wrong day: تسع عشرة not supported (source: سبع عشرة)",
    },
    {
        "id": "d_jannah_count",
        "answer": "عدد أبواب الجنة سبعون بابا (مصدر 1).",
        "chunk_ids": ["hadith_darimi_39619"],
        "expect": "drop",
        "why": "wrong number: سبعون not supported (source: ثمانية)",
    },
    {
        "id": "d_unrelated_claim",
        "answer": "غزوة بدر وقعت في القطب الشمالي (مصدر 1).",
        "chunk_ids": ["sira_646"],
        "expect": "drop",
        "why": "unrelated fabricated claim",
    },
    {
        "id": "d_invalid_citation",
        "answer": "أطلقت ناسا مسبارا إلى المريخ عام 2020 (مصدر 9).",
        "chunk_ids": ["sira_646", "fiqh_1184"],
        "expect": "drop",
        "why": "out-of-range citation; content unsupported by any chunk",
    },
    {
        "id": "d_khandaq_swap",
        "answer": "كانت غزوة الخندق يوم الجمعة في شهر رمضان (مصدر 1).",
        "chunk_ids": ["sira_646"],
        "expect": "drop",
        "why": (
            "subject swap: الخندق is not in the Badr passage (distinctive-token "
            "check), although كانت/الجمعة/رمضان all overlap"
        ),
    },
    {
        "id": "d_relation_inversion",
        "answer": "الذين يأكلون أموال اليتامى ظلما يدخلون الجنة (مصدر 1).",
        "chunk_ids": ["quran_4_10"],
        "expect": "drop",
        "why": (
            "relation inversion: shared vocabulary (أكل/أموال/يتامى) plus one "
            "unmatched distinctive verb (يدخلون) and الجنة must be rejected"
        ),
    },
    {
        "id": "d_negation_flip",
        "answer": "من اغتسل بعد الصلاة يعيد الغسل (مصدر 1).",
        "chunk_ids": ["fiqh_59"],
        "expect": "drop",
        "why": (
            "negation flip: the source says ولا يعيد الغسل; every matched word "
            "overlaps affirmatively and only the negation particle is absent, "
            "so coverage alone would accept it; the pole must reject it"
        ),
    },
    {
        "id": "d_multi_claim_location",
        "answer": "وقعت غزوة بدر يوم الجمعة في شهر رمضان في منطقة الأندلس (مصدر 1).",
        "chunk_ids": ["sira_646"],
        "expect": "drop",
        "why": (
            "two claims in one sentence: the Badr date part is fully supported "
            "and coverage stays above threshold, but the appended fabricated "
            "location (الأندلس, df=0) must reject the sentence"
        ),
    },
    {
        "id": "d_wrong_source_sentence",
        "answer": (
            "السنن الراتبة تشمل سنة الفجر والظهر والعصر والمغرب والعشاء (مصدر 1). "
            "والذين يأكلون أموال اليتامى لهم عذاب أليم (مصدر 1)."
        ),
        "chunk_ids": ["fiqh_160", "quran_4_10"],
        "expect": "keep",
        "absent": ["عذاب"],
        "why": (
            "citation adjacency: the second sentence cites مصدر 1 (fiqh_160) but its "
            "words come from the quran chunk -> sentence must be dropped even though "
            "the overall answer keeps the correct first sentence"
        ),
    },
]


def question_by_id(qid: str) -> dict:
    for q in EVAL_QUESTIONS:
        if q["id"] == qid:
            return q
    raise KeyError(qid)


def grounding_case_by_id(cid: str) -> dict:
    for c in GROUNDING_CASES:
        if c["id"] == cid:
            return c
    raise KeyError(cid)
