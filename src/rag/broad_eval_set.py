"""Broad, evidence-annotated evaluation set for Basira.

This set is deliberately separate from ``src/rag/eval_set.py`` so the original
22-question baseline remains comparable.  It covers the five corpus domains
(Quran, Tafsir, Hadith, Fiqh, Seerah), Arabic orthographic variants,
terminology ambiguity, and abstention.

Every ``evidence_ids`` entry was checked against the actual normalized chunk
text in ``data/processed/chunks.pkl`` before being added here.  The
``required_facts`` strings are human-readable acceptance criteria; the
``checks`` lists are the deterministic, normalized-substring rubrics used by
the live harness.  A check with ``optional=True`` is reported but does not
fail the question.

The set is intentionally *broad* rather than tailored to the current
retriever.  Several questions use natural wording that does not share the
corpus's exact lexical surface (e.g. "فرض" vs "كتب"), so the lexical-only
offline run is a lower bound and the live semantic run is the primary
measurement.
"""

from __future__ import annotations


def _chk(name: str, *, any_of=None, all_of=None, optional=False,
         kind="terms") -> dict:
    return {
        "name": name,
        "any_of": any_of or [],
        "all_of": all_of or [],
        "optional": optional,
        "kind": kind,
    }


BROAD_EVAL_QUESTIONS = [
    # ------------------------------------------------------------------
    # Quran (direct revelation text)
    # ------------------------------------------------------------------
    {
        "id": "quran_fasting",
        "category": "quran",
        "question": "لماذا فرض الله الصيام على المؤمنين؟",
        "answerable": True,
        "evidence_ids": ["quran_2_183", "tafsir_ibnkathir_2_183"],
        "required_facts": [
            "كتب الصيام على المؤمنين كما كتب على من قبلهم",
            "غاية الصيام التقوى (لعلكم تتقون)",
        ],
        "answer_criteria": "يذكر فرضية الصيام على الأمم السابقة وأن الغاية التقوى",
        "checks": [
            _chk("الصيام مكتوب على من قبلنا",
                 any_of=["كما كتب على الذين من قبلكم",
                         "كما كتب علي الذين من قبلكم",
                         "الذين من قبلكم",
                         "من قبلكم"]),
            _chk("غاية التقوى", any_of=["لعلكم تتقون", "تتقون", "التقوى"]),
        ],
        "notes": (
            "quran_2_183: 'كتب عليكم الصيام كما كتب على الذين من قبلكم لعلكم "
            "تتقون'.  tafsir_2_183 explains the same verse.  The user wording "
            "'فرض' does not occur in the Quran chunk, so this is a genuine "
            "lexical-to-semantic bridge."
        ),
    },
    {
        "id": "quran_wudu",
        "category": "quran",
        "question": "كيف يتوضأ المسلم قبل الصلاة؟",
        "answerable": True,
        "evidence_ids": ["quran_5_6", "fiqh_33"],
        "required_facts": [
            "غسل الوجه واليدين إلى المرافق",
            "مسح الرأس",
            "غسل الرجلين إلى الكعبين",
        ],
        "answer_criteria": "يذكر أعضاء الوضوء الأربعة وحدودها",
        "checks": [
            _chk("غسل الوجه", any_of=["اغسلوا وجوهكم", "غسل الوجه", "وجوهكم"]),
            _chk("الأيدي إلى المرافق",
                 any_of=["أيديكم إلى المرافق", "ايديكم الي المرافق",
                         "إلى المرافق", "الي المرافق"]),
            _chk("مسح الرأس",
                 any_of=["امسحوا برؤوسكم", "امسحوا بروسكم", "مسح الرأس",
                         "مسح الراس"]),
            _chk("الرجلين إلى الكعبين",
                 any_of=["أرجلكم إلى الكعبين", "ارجلكم الي الكعبين",
                         "إلى الكعبين", "الي الكعبين"]),
        ],
        "notes": (
            "quran_5_6 is the wudu verse; fiqh_33 quotes it and discusses "
            "the obligation.  Both contain the four limbs and their limits."
        ),
    },
    {
        "id": "quran_jumuah",
        "category": "quran",
        "question": "ماذا يجب على المسلم عند نداء صلاة الجمعة؟",
        "answerable": True,
        "evidence_ids": ["quran_62_9", "fiqh_255"],
        "required_facts": [
            "السعي إلى ذكر الله",
            "ترك البيع",
        ],
        "answer_criteria": "يذكر السعي إلى الذكر وترك البيع عند النداء",
        "checks": [
            _chk("السعي إلى ذكر الله",
                 any_of=["فاسعوا إلى ذكر الله", "فاسعوا الي ذكر الله",
                         "السعي إلى ذكر الله", "السعي الي ذكر الله",
                         "اسعوا"]),
            _chk("ترك البيع", any_of=["ذروا البيع", "ترك البيع", "وذروا البيع"]),
        ],
        "notes": (
            "quran_62_9 commands السعي إلى ذكر الله وترك البيع.  fiqh_255 "
            "quotes the same verse in the Friday-prayer chapter."
        ),
    },
    {
        "id": "quran_hajj",
        "category": "quran",
        "question": "على من يجب الحج إلى بيت الله الحرام؟",
        "answerable": True,
        "evidence_ids": ["quran_3_97", "fiqh_541"],
        "required_facts": ["الحج يجب على من استطاع إليه سبيلا"],
        "answer_criteria": "يذكر شرط الاستطاعة",
        "checks": [
            _chk("شرط الاستطاعة",
                 any_of=["من استطاع إليه سبيلا", "من استطاع اليه سبيلا",
                         "الاستطاعة", "الاستطاعه", "استطاع"]),
        ],
        "notes": (
            "quran_3_97: 'ولله على الناس حج البيت من استطاع إليه سبيلا'.  "
            "fiqh_541 lists the five conditions of obligation, including "
            "الاستطاعة."
        ),
    },
    {
        "id": "quran_riba",
        "category": "quran",
        "question": "ما حكم الربا في القرآن الكريم؟",
        "answerable": True,
        "evidence_ids": ["quran_2_275", "fiqh_1320"],
        "required_facts": ["الله أحل البيع وحرم الربا"],
        "answer_criteria": "يقرر تحريم الربا مع مقابلته بالبيع",
        "checks": [
            _chk("تحريم الربا",
                 any_of=["حرم الربا", "الربا محرم", "محرم", "حرام"]),
            _chk("إحلال البيع",
                 any_of=["أحل الله البيع", "احل الله البيع", "البيع حلال"]),
        ],
        "notes": (
            "quran_2_275: 'وأحل الله البيع وحرم الربا'.  fiqh_1320 defines "
            "الربا and states its prohibition."
        ),
    },
    {
        "id": "quran_parents",
        "category": "quran",
        "question": "كيف أمر الله بالإحسان إلى الوالدين؟",
        "answerable": True,
        "evidence_ids": ["quran_17_23", "quran_31_14"],
        "required_facts": [
            "عدم التأفف والنهر",
            "القول الكريم",
            "الشكر لله والوالدين",
        ],
        "answer_criteria": "يذكر الأدب مع الوالدين وعدم التأفف",
        "checks": [
            _chk("عدم التأفف",
                 any_of=["لا تقل لهما أف", "لا تقل لهما اف",
                         "التأفف", "التافف", "تأفف", "تافف"]),
            _chk("القول الكريم",
                 any_of=["قل لهما قولا كريما", "قول كريم", "قول كريما"]),
            _chk("الشكر لله والوالدين",
                 any_of=["اشكر لي ولوالديك", "اشكر لي ولولديك",
                         "الشكر لله والوالدين"], optional=True),
        ],
        "notes": (
            "quran_17_23 commands الإحسان and forbids أف/النهر; quran_31_14 "
            "adds الشكر.  The optional check is reported but not required."
        ),
    },
    {
        "id": "quran_inheritance",
        "category": "quran",
        "question": "ما نصيب البنت الواحدة من الميراث؟",
        "answerable": True,
        "evidence_ids": ["quran_4_11"],
        "required_facts": ["للبنت الواحدة النصف"],
        "answer_criteria": "يذكر النصف",
        "checks": [
            _chk("نصف البنت الواحدة", any_of=["النصف", "نصف"]),
        ],
        "notes": (
            "quran_4_11: 'وإن كانت واحدة فلها النصف'.  This is a precise "
            "inheritance fraction, not a general summary."
        ),
    },
    {
        "id": "quran_ayat_kursi",
        "category": "quran",
        "question": "ما أعظم آية في القرآن الكريم؟",
        "answerable": True,
        "evidence_ids": ["tafsir_ibnkathir_2_255", "quran_2_255"],
        "required_facts": ["آية الكرسي هي أعظم آية"],
        "answer_criteria": "يحدد آية الكرسي",
        "checks": [
            _chk("آية الكرسي",
                 any_of=["آية الكرسي", "ايه الكرسي", "الكرسي"]),
            _chk("أفضل آية", any_of=["أفضل آية", "افضل ايه", "أعظم آية",
                                     "اعظم ايه"], optional=True),
        ],
        "notes": (
            "tafsir_2_255 opens: 'هذه آية الكرسي ولها شأن عظيم قد صح الحديث "
            "عن رسول الله ﷺ بأنها أفضل آية في كتاب الله'.  quran_2_255 is "
            "the verse itself."
        ),
    },
    # ------------------------------------------------------------------
    # Tafsir (explicit commentary)
    # ------------------------------------------------------------------
    {
        "id": "tafsir_fatiha",
        "category": "tafsir",
        "question": "ما معنى الحمد لله رب العالمين؟",
        "answerable": True,
        "evidence_ids": ["tafsir_ibnkathir_1_2", "quran_1_2"],
        "required_facts": ["الحمد هو الشكر لله خالصا", "الله رب العالمين"],
        "answer_criteria": "يفسر الحمد بالشكر/الثناء لله",
        "checks": [
            _chk("الحمد شكر لله",
                 any_of=["الشكر لله", "شكر لله", "الثناء على الله",
                         "الثناء لله", "الحمد هو الشكر"]),
            _chk("رب العالمين",
                 any_of=["رب العالمين", "رب العالمين"], optional=True),
        ],
        "notes": (
            "tafsir_1_2 quotes ابن جرير: 'معنى الحمد لله الشكر لله خالصا "
            "دون سائر المخلوقين'.  quran_1_2 is the verse."
        ),
    },
    {
        "id": "tafsir_fasting",
        "category": "tafsir",
        "question": "ما معنى قوله تعالى كتب عليكم الصيام؟",
        "answerable": True,
        "evidence_ids": ["tafsir_ibnkathir_2_183", "quran_2_183"],
        "required_facts": [
            "الصيام إمساك عن الطعام والشراب والوقاع",
            "بنية خالصة لله",
        ],
        "answer_criteria": "يعرّف الصيام بالإمساك والنية",
        "checks": [
            _chk("الإمساك عن الطعام والشراب",
                 any_of=["الإمساك عن الطعام والشراب", "الامساك عن الطعام والشراب",
                         "الإمساك", "الامساك"]),
            _chk("النية الخالصة",
                 any_of=["نية خالصة", "نيه خالصه", "بنية خالصة",
                         "بنيه خالصه", "نية"]),
        ],
        "notes": (
            "tafsir_2_183: 'وهو الإمساك عن الطعام والشراب والوقاع بنية خالصة "
            "لله'.  quran_2_183 is the verse."
        ),
    },
    {
        "id": "tafsir_hajj",
        "category": "tafsir",
        "question": "ما تفسير آية حج البيت من استطاع إليه سبيلا؟",
        "answerable": True,
        "evidence_ids": ["tafsir_ibnkathir_3_97", "quran_3_97"],
        "required_facts": [
            "أول بيت وضع للناس هو الكعبة/بكة",
            "الحج على المستطيع",
        ],
        "answer_criteria": "يذكر أول بيت وشرط الاستطاعة",
        "checks": [
            _chk("أول بيت وضع للناس",
                 any_of=["أول بيت وضع للناس", "اول بيت وضع للناس",
                         "أول بيت", "اول بيت"]),
            _chk("الاستطاعة",
                 any_of=["الاستطاعة", "الاستطاعه", "استطاع"]),
        ],
        "notes": (
            "tafsir_3_97 explains 'أول بيت وضع للناس' and the Hajj obligation; "
            "quran_3_97 contains 'من استطاع إليه سبيلا'."
        ),
    },
    {
        "id": "tafsir_ikhlas",
        "category": "tafsir",
        "question": "ما تفسير سورة الإخلاص الآية الأولى؟",
        "answerable": True,
        "evidence_ids": ["tafsir_ibnkathir_112_1"],
        "required_facts": [
            "قل هو الله أحد",
            "سبب النزول سؤال المشركين عن نسب الرب",
        ],
        "answer_criteria": "يذكر معنى الآية وسؤال المشركين",
        "checks": [
            _chk("قل هو الله أحد",
                 any_of=["قل هو الله أحد", "قل هو الله احد"]),
            _chk("سبب النزول",
                 any_of=["المشركون", "المشركين", "انسب لنا ربك",
                         "نسب ربك"], optional=True),
        ],
        "notes": (
            "tafsir_112_1 contains the verse and the story that the polytheists "
            "asked the Prophet to describe his Lord's lineage."
        ),
    },
    # ------------------------------------------------------------------
    # Hadith (primary reports)
    # ------------------------------------------------------------------
    {
        "id": "hadith_pillars",
        "category": "hadith",
        "question": "على كم شيء بني الإسلام؟",
        "answerable": True,
        "evidence_ids": ["hadith_bukhari_8"],
        "required_facts": ["بني الإسلام على خمس"],
        "answer_criteria": "يذكر الخمس",
        "checks": [
            _chk("خمس", any_of=["خمس", "خمسة", "خمسه"]),
        ],
        "notes": (
            "hadith_bukhari_8: 'بني الإسلام على خمس: شهادة أن لا إله إلا الله "
            "وأن محمدا رسول الله، وإقام الصلاة، وإيتاء الزكاة، والحج، وصوم "
            "رمضان'."
        ),
    },
    {
        "id": "hadith_names",
        "category": "hadith",
        "question": "كم عدد أسماء الله الحسنى؟",
        "answerable": True,
        "evidence_ids": ["hadith_bukhari_2624"],
        "required_facts": ["تسعة وتسعون اسما"],
        "answer_criteria": "يذكر 99",
        "checks": [
            _chk("تسعة وتسعون",
                 any_of=["تسعة وتسعين", "تسعه وتسعين", "تسعة وتسعون",
                         "تسعه وتسعون", "99"]),
        ],
        "notes": (
            "hadith_bukhari_2624: 'إن لله تسعة وتسعين اسما مائة إلا واحدا "
            "من أحصاها دخل الجنة'."
        ),
    },
    {
        "id": "hadith_gheebah",
        "category": "hadith",
        "question": "ما تعريف الغيبة؟",
        "answerable": True,
        "evidence_ids": ["hadith_muslim_13708", "fiqh_512"],
        "required_facts": ["ذكرك أخاك بما يكره"],
        "answer_criteria": "يعرّفها بذكر المسلم بما يكره",
        "checks": [
            _chk("ذكرك أخاك بما يكره",
                 any_of=["ذكرك أخاك بما يكره", "ذكرك اخاك بما يكره",
                         "بما يكره", "ذكر أخاك بما يكره"]),
        ],
        "notes": (
            "hadith_muslim_13708: 'أتدرو ن ما الغيبة؟ ... ذكرك أخاك بما يكره'.  "
            "fiqh_512 discusses the expiation, not the definition, but is a "
            "useful secondary source."
        ),
    },
    {
        "id": "hadith_zakat_fitr",
        "category": "hadith",
        "question": "ما مقدار زكاة الفطر؟",
        "answerable": True,
        "evidence_ids": ["hadith_bukhari_1454", "fiqh_346"],
        "required_facts": ["صاع من طعام/تمر/شعير"],
        "answer_criteria": "يذكر الصاع",
        "checks": [
            _chk("صاع", any_of=["صاع"]),
            _chk("من طعام أو تمر أو شعير",
                 any_of=["من طعام", "من تمر", "من شعير", "من أقط",
                         "من زبيب"], optional=True),
        ],
        "notes": (
            "hadith_bukhari_1454: 'كنا نخرج زكاة الفطر صاعا من طعام، أو صاعا "
            "من شعير، أو صاعا من تمر، أو صاعا من أقط، أو صاعا من زبيب'.  "
            "fiqh_346 states the same amount."
        ),
    },
    {
        "id": "hadith_tayammum",
        "category": "hadith",
        "question": "كيف شرع التيمم في سفر عائشة؟",
        "answerable": True,
        "evidence_ids": ["hadith_bukhari_330", "fiqh_66"],
        "required_facts": [
            "انقطع عقد عائشة فحبس الناس",
            "لم يكن معهم ماء",
            "أنزل الله آية التيمم",
        ],
        "answer_criteria": "يروي القصة وسبب نزول آية التيمم",
        "checks": [
            _chk("انقطع العقد",
                 any_of=["انقطع عقد", "انقطع العقد", "عقد لي",
                         "سقط عقد"]),
            _chk("عدم وجود الماء",
                 any_of=["ليسوا على ماء", "ليس معهم ماء", "لم يجدوا ماء",
                         "على غير ماء", "لم يكن معهم ماء"]),
            _chk("آية التيمم",
                 any_of=["آية التيمم", "ايه التيمم", "فأنزل الله آية التيمم",
                         "فانزل الله ايه التيمم"]),
        ],
        "notes": (
            "hadith_bukhari_330 ends: 'فأنزل الله آية التيمم فتيمموا'.  "
            "fiqh_66 gives the legal definition and Quranic basis."
        ),
    },
    {
        "id": "hadith_khuff",
        "category": "hadith",
        "question": "ما حكم المسح على الخفين؟",
        "answerable": True,
        "evidence_ids": ["hadith_bukhari_202", "fiqh_49"],
        "required_facts": ["المسح على الخفين جائز/مشروع", "فعله النبي"],
        "answer_criteria": "يقرر الجواز ويسند إلى فعل النبي",
        "checks": [
            _chk("الجواز", any_of=["جائز", "يجوز", "مشروع", "ثابت"]),
            _chk("فعل النبي",
                 any_of=["مسح على الخفين", "مسح على خفيه", "مسح النبي",
                         "رأيت رسول الله"]),
        ],
        "notes": (
            "hadith_bukhari_202: 'أن النبي ﷺ مسح على الخفين'.  fiqh_49 "
            "documents the consensus and the many narrations."
        ),
    },
    {
        "id": "hadith_age",
        "category": "hadith",
        "question": "كم كان عمر النبي صلى الله عليه وسلم حين توفي؟",
        "answerable": True,
        "evidence_ids": ["hadith_bukhari_3386"],
        "required_facts": ["ثلاث وستون سنة"],
        "answer_criteria": "يذكر 63",
        "checks": [
            _chk("ثلاث وستون",
                 any_of=["ثلاث وستين", "ثلاث وستون", "ثلاثه وستين",
                         "ثلاثة وستون", "63"]),
        ],
        "notes": (
            "hadith_bukhari_3386: 'أن النبي ﷺ توفي وهو ابن ثلاث وستين'."
        ),
    },
    {
        "id": "hadith_revelation",
        "category": "hadith",
        "question": "كيف بدأ الوحي إلى النبي صلى الله عليه وسلم؟",
        "answerable": True,
        "evidence_ids": ["hadith_bukhari_3"],
        "required_facts": [
            "بدأ بالرؤيا الصالحة",
            "ثم الخلوة في غار حراء",
            "ثم نزول اقرأ",
        ],
        "answer_criteria": "يذكر الرؤيا وحراء وبداية الوحي",
        "checks": [
            _chk("الرؤيا الصالحة",
                 any_of=["الرؤيا الصالحة", "الرؤيا الصالحه", "الرؤيا"]),
            _chk("غار حراء", any_of=["غار حراء", "غار حرا", "حراء"]),
            _chk("اقرأ", any_of=["اقرأ", "اقرا"], optional=True),
        ],
        "notes": (
            "hadith_bukhari_3 is the first hadith in Bukhari: 'أول ما بدئ به "
            "رسول الله ﷺ من الوحي الرؤيا الصالحة في النوم ... وكان يخلو بغار "
            "حراء ...'."
        ),
    },
    {
        "id": "hadith_travel",
        "category": "hadith",
        "question": "كم يصلي المسافر؟",
        "answerable": True,
        "evidence_ids": ["hadith_muslim_8762", "fiqh_241"],
        "required_facts": ["يصلي ركعتين"],
        "answer_criteria": "يذكر القصر إلى ركعتين",
        "checks": [
            _chk("ركعتين", any_of=["ركعتين", "ركعتان", "ركعتين"]),
        ],
        "notes": (
            "hadith_muslim_8762: 'أنه صلى صلاة المسافر بمنى وغيرها ركعتين'.  "
            "fiqh_241 discusses where the shortening begins."
        ),
    },
    {
        "id": "hadith_itikaf",
        "category": "hadith",
        "question": "متى كان النبي صلى الله عليه وسلم يعتكف؟",
        "answerable": True,
        "evidence_ids": ["hadith_muslim_9946", "fiqh_396"],
        "required_facts": ["في العشر الأواخر من رمضان"],
        "answer_criteria": "يذكر العشر الأواخر",
        "checks": [
            _chk("العشر الأواخر",
                 any_of=["العشر الأواخر", "العشر الاواخر", "عشر الأواخر",
                         "عشر الاواخر"]),
        ],
        "notes": (
            "hadith_muslim_9946: 'أراد الاعتكاف في العشر الأواخر من رمضان'.  "
            "fiqh_396 adds that he used to observe it every Ramadan."
        ),
    },
    # ------------------------------------------------------------------
    # Fiqh (derived rulings)
    # ------------------------------------------------------------------
    {
        "id": "fiqh_zakat_fitr",
        "category": "fiqh",
        "question": "ما حكم زكاة الفطر ومتى تجب؟",
        "answerable": True,
        "evidence_ids": ["fiqh_346", "fiqh_347", "hadith_bukhari_1454"],
        "required_facts": [
            "واجبة على كل مسلم",
            "مقدارها صاع من طعام",
            "تؤدى قبل صلاة العيد",
        ],
        "answer_criteria": "يجمع بين الوجوب والمقدار والوقت",
        "checks": [
            _chk("الوجوب", any_of=["واجبة", "واجبه", "فرض", "تجب"]),
            _chk("الصاع", any_of=["صاع"]),
            _chk("قبل صلاة العيد",
                 any_of=["قبل الصلاة", "قبل صلاة العيد", "قبل خروج الناس",
                         "قبل العيد"]),
        ],
        "notes": (
            "fiqh_346: 'وهي واجبة على كل فرد من المسلمين ... صاعا من تمر أو "
            "صاعا من شعير'.  fiqh_347 discusses the time and quotes 'أمرنا "
            "رسول الله ﷺ بزكاة الفطر أن تؤدى قبل خروج الناس إلى الصلاة'."
        ),
    },
    {
        "id": "fiqh_wudu",
        "category": "fiqh",
        "question": "ما فرائض الوضوء؟",
        "answerable": True,
        "evidence_ids": ["fiqh_33", "quran_5_6"],
        "required_facts": [
            "غسل الوجه واليدين إلى المرافق",
            "مسح الرأس",
            "غسل الرجلين إلى الكعبين",
        ],
        "answer_criteria": "يعدد الأعضاء الأربعة",
        "checks": [
            _chk("الوجه", any_of=["الوجه", "وجوهكم"]),
            _chk("اليدين", any_of=["اليدين", "أيديكم", "ايديكم"]),
            _chk("الرأس", any_of=["الرأس", "الراس", "برؤوسكم", "بروسكم"]),
            _chk("الرجلين", any_of=["الرجلين", "أرجلكم", "ارجلكم"]),
        ],
        "notes": (
            "fiqh_33 quotes the Quranic verse and treats these four as the "
            "obligatory acts; quran_5_6 is the primary text."
        ),
    },
    {
        "id": "fiqh_tayammum",
        "category": "fiqh",
        "question": "متى يجوز التيمم؟",
        "answerable": True,
        "evidence_ids": ["fiqh_66", "fiqh_67"],
        "required_facts": [
            "عند عدم الماء أو عدم كفايته",
            "عند المرض أو الجراحة",
        ],
        "answer_criteria": "يذكر السببين: فقد الماء والمرض",
        "checks": [
            _chk("عدم الماء",
                 any_of=["لم يجد الماء", "عدم الماء", "عدم وجود الماء",
                         "لا يجد الماء", "لا يكفيه"]),
            _chk("المرض أو الجراحة",
                 any_of=["مرض", "جراحة", "جراحه", "جرح"]),
        ],
        "notes": (
            "fiqh_67: 'يباح التيمم ... إذا لم يجد الماء، أو وجد منه ما لا "
            "يكفيه ... إذا كان به جراحة أو مرض'."
        ),
    },
    {
        "id": "fiqh_travel",
        "category": "fiqh",
        "question": "متى يقصر المسافر الصلاة؟",
        "answerable": True,
        "evidence_ids": ["fiqh_241", "hadith_muslim_8762"],
        "required_facts": [
            "بمفارقة الحضر/الخروج من البلد",
            "يصلي ركعتين",
        ],
        "answer_criteria": "يذكر بداية القصر وعدد الركعات",
        "checks": [
            _chk("مفارقة الحضر",
                 any_of=["مفارقة الحضر", "مفارقه الحضر", "الخروج من البلد",
                         "بعد خروجه من المدينة"]),
            _chk("ركعتين", any_of=["ركعتين", "ركعتان"]),
        ],
        "notes": (
            "fiqh_241: 'قصر الصلاة يشرع بمفارقة الحضر والخروج من البلد'.  "
            "hadith_muslim_8762 records the Prophet praying two rak'ahs as a "
            "traveler."
        ),
    },
    {
        "id": "fiqh_gheebah",
        "category": "fiqh",
        "question": "ما كفارة الغيبة؟",
        "answerable": True,
        "evidence_ids": ["fiqh_512", "hadith_muslim_13708"],
        "required_facts": ["الاستغفار لمن اغتبته"],
        "answer_criteria": "يذكر الاستغفار للمغتاب",
        "checks": [
            _chk("الاستغفار",
                 any_of=["الاستغفار", "استغفر", "تستغفر", "اغفر"]),
        ],
        "notes": (
            "fiqh_512: 'إن كفارة الغيبة أن تستغفر لمن اغتبته، تقول اللهم اغفر "
            "لنا وله'."
        ),
    },
    {
        "id": "fiqh_riba",
        "category": "fiqh",
        "question": "ما الحكمة من تحريم الربا؟",
        "answerable": True,
        "evidence_ids": ["fiqh_1320", "fiqh_1321", "quran_2_275"],
        "required_facts": [
            "الربا ظلم/زيادة على رأس المال",
            "يسبب العداوة ويقضي على التعاون",
        ],
        "answer_criteria": "يذكر الظلم والضرر الاجتماعي",
        "checks": [
            _chk("الظلم", any_of=["ظلم", "الظلم",
                                    "زيادة على رأس المال",
                                    "الزيادة على رأس المال"]),
            _chk("العداوة", any_of=["عداوة", "عداوه", "العداوة"]),
            _chk("الضرر", any_of=["ضرر", "الضرر"], optional=True),
        ],
        "notes": (
            "fiqh_1320 defines الربا as زيادة على رأس المال and says it is "
            "forbidden in all revealed religions; fiqh_1321 lists the harms: "
            "العداوة, cutting cooperation, etc."
        ),
    },
    {
        "id": "fiqh_masah",
        "category": "fiqh",
        "question": "ما مدة المسح على الخفين؟",
        "answerable": True,
        "evidence_ids": ["fiqh_51"],
        "required_facts": [
            "للمقيم يوم وليلة",
            "للمسافر ثلاثة أيام ولياليها",
        ],
        "answer_criteria": "يذكر المدتين",
        "checks": [
            _chk("يوم وليلة للمقيم",
                 any_of=["يوم وليلة", "يوم وليله", "يوما وليلة",
                         "يوما وليله"]),
            _chk("ثلاثة أيام للمسافر",
                 any_of=["ثلاثة أيام", "ثلاثه ايام", "ثلاثا", "ثلاثة أيام ولياليها",
                         "ثلاثه ايام ولياليها"]),
        ],
        "notes": (
            "fiqh_51: 'مدة المسح على الخفين للمقيم يوم وليلة، وللمسافر ثلاثة "
            "أيام ولياليها'."
        ),
    },
    {
        "id": "fiqh_hajj",
        "category": "fiqh",
        "question": "ما شروط وجوب الحج؟",
        "answerable": True,
        "evidence_ids": ["fiqh_541", "fiqh_542", "quran_3_97"],
        "required_facts": [
            "الإسلام والبلوغ والعقل والحرية والاستطاعة",
        ],
        "answer_criteria": "يعدد الشروط الخمسة",
        "checks": [
            _chk("الإسلام", any_of=["الإسلام", "الاسلام"]),
            _chk("البلوغ", any_of=["البلوغ", "البلوغ"]),
            _chk("العقل", any_of=["العقل"]),
            _chk("الحرية", any_of=["الحرية", "الحريه"]),
            _chk("الاستطاعة", any_of=["الاستطاعة", "الاستطاعه"]),
        ],
        "notes": (
            "fiqh_541: 'اتفق الفقهاء على أنه يشترط لوجوب الحج ... الإسلام، "
            "البلوغ، العقل، الحرية، الاستطاعة'.  quran_3_97 supplies the "
            "scriptural basis for الاستطاعة."
        ),
    },
    # ------------------------------------------------------------------
    # Seerah (prophetic biography)
    # ------------------------------------------------------------------
    {
        "id": "sira_birth",
        "category": "seerah",
        "question": "متى ولد النبي صلى الله عليه وسلم؟",
        "answerable": True,
        "evidence_ids": ["sira_157"],
        "required_facts": [
            "يوم الاثنين",
            "الثاني عشر من ربيع الأول",
            "عام الفيل",
        ],
        "answer_criteria": "يذكر اليوم والشهر وعام الفيل",
        "checks": [
            _chk("يوم الاثنين", any_of=["يوم الاثنين", "يوم الاثنين"]),
            _chk("ربيع الأول", any_of=["ربيع الأول", "ربيع الاول"]),
            _chk("عام الفيل", any_of=["عام الفيل", "عام الفيل"]),
        ],
        "notes": (
            "sira_157: 'ولد رسول الله ﷺ يوم الاثنين، لاثنتي عشرة ليلة خلت من "
            "شهر ربيع الأول، عام الفيل'."
        ),
    },
    {
        "id": "sira_foster",
        "category": "seerah",
        "question": "من أرضعت النبي صلى الله عليه وسلم؟",
        "answerable": True,
        "evidence_ids": ["sira_159"],
        "required_facts": ["حليمة بنت أبي ذؤيب"],
        "answer_criteria": "يذكر حليمة",
        "checks": [
            _chk("حليمة", any_of=["حليمة", "حليمه"]),
        ],
        "notes": (
            "sira_159 names 'حليمة بنت أبي ذؤيب عبد الله بن الحارث' as the "
            "foster-mother."
        ),
    },
    {
        "id": "sira_abyssinia",
        "category": "seerah",
        "question": "أين كانت الهجرة الأولى للمسلمين؟",
        "answerable": True,
        "evidence_ids": ["sira_312"],
        "required_facts": ["إلى أرض الحبشة"],
        "answer_criteria": "يذكر الحبشة",
        "checks": [
            _chk("الحبشة", any_of=["الحبشة", "الحبشه"]),
        ],
        "notes": (
            "sira_312: 'لو خرجتم إلى أرض الحبشة فإن بها ملكا لا يظلم عنده "
            "أحد'."
        ),
    },
    {
        "id": "sira_aqaba",
        "category": "seerah",
        "question": "كم كان عدد من بايع النبي في بيعة العقبة الثانية؟",
        "answerable": True,
        "evidence_ids": ["sira_440"],
        "required_facts": ["ثلاثة وسبعون رجلا", "وامرأتان"],
        "answer_criteria": "يذكر 73 رجلا وامرأتين",
        "checks": [
            _chk("ثلاثة وسبعون رجلا",
                 any_of=["ثلاثة وسبعون", "ثلاثه وسبعون", "73"]),
            _chk("امرأتان", any_of=["امرأتان", "امراتان", "امرأتين",
                                     "امرأتين"], optional=True),
        ],
        "notes": (
            "sira_440: 'ونحن ثلاثة وسبعون رجلا، ومعه امرأتان من نسائنا'."
        ),
    },
    {
        "id": "sira_hudaybiyah",
        "category": "seerah",
        "question": "في أي سنة كانت صلح الحديبية؟",
        "answerable": True,
        "evidence_ids": ["sira_1026", "hadith_bukhari_1633"],
        "required_facts": ["في السنة السادسة"],
        "answer_criteria": "يذكر السنة السادسة",
        "checks": [
            _chk("السنة السادسة",
                 any_of=["سنة ست", "سنه ست", "السادسة", "السادسه",
                         "6 هجريه", "6"]),
        ],
        "notes": (
            "sira_1026 text opens: 'أمر الحديبية في آخر سنة ست'.  "
            "hadith_bukhari_1633 places the event 'زمن الحديبية'."
        ),
    },
    {
        "id": "sira_mecca",
        "category": "seerah",
        "question": "في أي سنة كان فتح مكة؟",
        "answerable": True,
        "evidence_ids": ["sira_1115"],
        "required_facts": ["في رمضان سنة ثمان"],
        "answer_criteria": "يذكر السنة الثامنة",
        "checks": [
            _chk("السنة الثامنة",
                 any_of=["سنة ثمان", "سنه ثمان", "الثامنة", "الثامنه",
                         "8 هجريه", "8"]),
        ],
        "notes": (
            "sira_1115 text opens: 'ذكر فتح مكة في شهر رمضان سنة ثمان'."
        ),
    },
    {
        "id": "sira_tabuk",
        "category": "seerah",
        "question": "في أي سنة كانت غزوة تبوك؟",
        "answerable": True,
        "evidence_ids": ["sira_1228"],
        "required_facts": ["في رجب سنة تسع"],
        "answer_criteria": "يذكر السنة التاسعة",
        "checks": [
            _chk("السنة التاسعة",
                 any_of=["سنة تسع", "سنه تسع", "التاسعة", "التاسعه",
                         "9 هجريه", "9"]),
        ],
        "notes": (
            "sira_1228 text opens: 'غزوة تبوك في رجب سنة تسع'."
        ),
    },
    {
        "id": "sira_farewell",
        "category": "seerah",
        "question": "ماذا أوصى النبي في خطبة حجة الوداع؟",
        "answerable": True,
        "evidence_ids": ["sira_1338"],
        "required_facts": [
            "حرمة الدماء والأموال",
            "أداء الأمانات",
            "وضع الربا",
        ],
        "answer_criteria": "يذكر حرمة الدماء والأموال ووصايا الخطبة",
        "checks": [
            _chk("حرمة الدماء والأموال",
                 any_of=["دماءكم وأموالكم", "الدماء والأموال",
                         "حرمة الدماء", "حرام إلى أن تلقوا ربكم"]),
            _chk("أداء الأمانة",
                 any_of=["الأمانة", "الامانه", "فليؤدها"], optional=True),
        ],
        "notes": (
            "sira_1338 quotes the farewell sermon: 'إن دماءكم وأموالكم عليكم "
            "حرام ... فمن كانت عنده أمانة فليؤدها ... وإن كل ربا موضوع'."
        ),
    },
    {
        "id": "sira_isra",
        "category": "seerah",
        "question": "ما هي رحلة الإسراء والمعراج؟",
        "answerable": True,
        "evidence_ids": ["sira_390"],
        "required_facts": [
            "الإسراء من المسجد الحرام إلى المسجد الأقصى",
            "المعراج إلى السماء",
        ],
        "answer_criteria": "يذكر الإسراء إلى بيت المقدس والمعراج",
        "checks": [
            _chk("من المسجد الحرام إلى المسجد الأقصى",
                 any_of=["المسجد الحرام", "المسجد الأقصى", "بيت المقدس",
                         "بيت المقدس"]),
            _chk("المعراج", any_of=["المعراج", "السماء", "السماء"],
                 optional=True),
        ],
        "notes": (
            "sira_390: 'ثم أسري برسول الله ﷺ من المسجد الحرام إلى المسجد "
            "الأقصى، وهو بيت المقدس'."
        ),
    },
    # ------------------------------------------------------------------
    # Arabic orthographic / spelling variants
    # ------------------------------------------------------------------
    {
        "id": "variant_witr",
        "category": "arabic_variant",
        "question": "ما حكم صلوة الوتر؟",
        "answerable": True,
        "evidence_ids": ["fiqh_165"],
        "required_facts": ["الوتر سنة مؤكدة"],
        "answer_criteria": "يقرر أن الوتر سنة مؤكدة",
        "checks": [
            _chk("الوتر سنة مؤكدة",
                 any_of=["سنة مؤكدة", "سنه مؤكده", "سنة مؤكده",
                         "سنه مؤكدة"]),
        ],
        "notes": (
            "Deliberate spelling variants: صلوة (ة→ه in normalize) and الوتر.  "
            "fiqh_165 contains 'الوتر سنة مؤكدة'."
        ),
    },
    {
        "id": "variant_jannah",
        "category": "arabic_variant",
        "question": "كم عدد ابواب الجنه؟",
        "answerable": True,
        "evidence_ids": ["hadith_darimi_39619"],
        "required_facts": ["ثمانية أبواب"],
        "answer_criteria": "يذكر ثمانية",
        "checks": [
            _chk("ثمانية أبواب",
                 any_of=["ثمانية", "ثمانيه", "ثمانية أبواب",
                         "ثمانيه ابواب"]),
        ],
        "notes": (
            "hadith_darimi_39619: 'للجنة ثمانية أبواب'.  The query omits "
            "hamzas and uses colloquial spelling."
        ),
    },
    {
        "id": "variant_orphans",
        "category": "arabic_variant",
        "question": "ما جزاء الذين ياكلون اموال اليتامي ظلما؟",
        "answerable": True,
        "evidence_ids": ["quran_4_10"],
        "required_facts": ["يأكلون في بطونهم نارا", "سيصلون سعيرا"],
        "answer_criteria": "يذكر وعيد النار",
        "checks": [
            _chk("النار في البطون",
                 any_of=["بطونهم نارا", "في بطونهم نار", "نارا"]),
            _chk("سعير", any_of=["سعيرا", "سعير"]),
        ],
        "notes": (
            "quran_4_10.  The query uses ياكلون/اموال/اليتامي/ظلما without "
            "hamzas or alef maqsura, exercising the lexical folding path."
        ),
    },
    # ------------------------------------------------------------------
    # Ambiguity (manual review)
    # ------------------------------------------------------------------
    {
        "id": "ambiguous_sunan",
        "category": "ambiguous",
        "question": "ما هي السنن المؤكدة؟",
        "answerable": True,
        "evidence_ids": [
            "fiqh_160", "fiqh_164", "fiqh_165", "fiqh_195", "fiqh_184",
            "fiqh_1477",
        ],
        "required_facts": [],
        "answer_criteria": (
            "إجابة مؤهلة: طلب توضيح أو بيان أن المصطلح يشمل أكثر من باب "
            "(رواتب الصلاة، الوتر، الجماعة، الكسوف، العقيقة...)"
        ),
        "checks": [
            _chk("تصنيف السؤال الغامض", kind="ambiguous"),
        ],
        "notes": (
            "Corpus inspection: سنة مؤكدة is used for رواتب الصلاة (fiqh_160), "
            "الوتر (fiqh_165), الجماعة (fiqh_195), الكسوف (fiqh_184), "
            "العقيقة (fiqh_1477).  There is no single correct list; this row "
            "requires manual review of whether the answer qualifies the term."
        ),
    },
    # ------------------------------------------------------------------
    # Unanswerable / abstention
    # ------------------------------------------------------------------
    {
        "id": "unanswerable_planet",
        "category": "unanswerable",
        "question": "ما هو أكبر كوكب في المجموعة الشمسية؟",
        "answerable": False,
        "evidence_ids": [],
        "required_facts": [],
        "answer_criteria": "رفض: لا توجد أدلة في المصادر",
        "checks": [],
        "notes": "Modern astronomy; absent from all five corpora.",
    },
    {
        "id": "unanswerable_gold",
        "category": "unanswerable",
        "question": "ما هو سعر الذهب اليوم؟",
        "answerable": False,
        "evidence_ids": [],
        "required_facts": [],
        "answer_criteria": "رفض: لا توجد أدلة في المصادر",
        "checks": [],
        "notes": "Time-sensitive market price; absent from the corpus.",
    },
    {
        "id": "unanswerable_python",
        "category": "unanswerable",
        "question": "كيف أستخدم بايثون لتحليل البيانات؟",
        "answerable": False,
        "evidence_ids": [],
        "required_facts": [],
        "answer_criteria": "رفض: لا توجد أدلة في المصادر",
        "checks": [],
        "notes": "Modern programming topic; clearly off-domain.",
    },
    {
        "id": "unanswerable_driving",
        "category": "unanswerable",
        "question": "ما هي شروط استخراج رخصة القيادة في السعودية؟",
        "answerable": False,
        "evidence_ids": [],
        "required_facts": [],
        "answer_criteria": "رفض: لا توجد أدلة في المصادر",
        "checks": [],
        "notes": (
            "Modern Saudi traffic regulation; looks in-domain (fiqh-like) but "
            "is absent, so it probes gate precision."
        ),
    },
]


BROAD_RUBRICS = {q["id"]: q.get("checks", []) for q in BROAD_EVAL_QUESTIONS}
