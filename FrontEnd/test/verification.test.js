import { test } from "node:test";
import assert from "node:assert/strict";

import { verificationView } from "../src/utils/verification.js";

const PARTIAL = {
  verification_status: "partial",
  status_label: "تم التحقق — اقتباس جزئي",
  status_explanation: "النص جزء حرفي من الآية.",
  input_text: "ولا تقربوا الصلاة",
  input_explanation: "",
  text_status: "صحيح",
  is_truncated: true,
  full_text: "يَا أَيُّهَا الَّذِينَ آمَنُوا لَا تَقْرَبُوا الصَّلَاةَ...",
  source: "القرآن الكريم — النساء:43",
  danger_level: "متوسطة",
  danger_level_applicable: true,
  danger_level_meaning: "احتمال الفهم المضلل، وليست درجة ثقة.",
  quotation: {
    found: true,
    match_type: "partial",
    is_partial: true,
    is_complete: false,
    canonical_text: "يَا أَيُّهَا الَّذِينَ آمَنُوا لَا تَقْرَبُوا الصَّلَاةَ...",
    reference: "النساء:43",
    source: "quran",
  },
  interpretation: {
    provided: false,
    status: "not_provided",
    label: "لم يتم تقديم تفسير لتقييمه",
  },
  evidence: [
    { source: "quran", reference: "النساء:43", text: "الآية", role: "canonical" },
    { source: "tafsir", reference: "تفسير ابن كثير 4:43", text: "التفسير", role: "interpretation" },
  ],
};

test("partial quotation is displayed as partial and keeps the canonical text", () => {
  const view = verificationView(PARTIAL);
  assert.equal(view.status, "partial");
  assert.equal(view.matchLabel, "اقتباس جزئي");
  assert.equal(view.completenessLabel, "الاقتباس جزئي");
  assert.equal(view.canonical, PARTIAL.full_text);
  assert.equal(view.reference, "النساء:43");
  assert.equal(view.isPartial, true);
  assert.equal(view.isComplete, false);
});

test("no supplied interpretation uses the explicit not-provided label", () => {
  const view = verificationView(PARTIAL);
  assert.equal(view.interpretationProvided, false);
  assert.equal(
    view.interpretationLabel,
    "لم يتم تقديم تفسير لتقييمه",
  );
  assert.equal(view.interpretationStatus, "not_provided");
});

test("contradicted interpretation is separate from the verified quotation", () => {
  const view = verificationView({
    ...PARTIAL,
    verification_status: "partial",
    status_label: "تم التحقق — اقتباس جزئي — التفسير غير مدعوم",
    input_explanation: "الصلاة محرمة.",
    interpretation: {
      provided: true,
      status: "contradicted",
      label: "التفسير غير مدعوم أو يخالف المصادر",
      assessment: "الآية تتحدث عن حال السكر.",
      correction: "النهي متعلق بالصلاة في حال السكر.",
    },
  });
  assert.equal(view.quotationFound, true);
  assert.equal(view.interpretationStatus, "contradicted");
  assert.equal(view.interpretationTone, "danger");
  assert.equal(view.matchLabel, "اقتباس جزئي");
  assert.match(view.statusLabel || view.label, /اقتباس جزئي/);
});

test("service error is never rendered as a missing quotation", () => {
  const view = verificationView({
    verification_status: "error",
    status_label: "تعذّر إكمال التحقق",
    status_explanation: "حدث خطأ أثناء البحث.",
    verification_error: "تعذّر البحث في المصادر.",
    input_text: "ولا تقربوا الصلاة",
    text_status: "خطأ في التحقق",
    danger_level: "غير محسوبة",
    danger_level_applicable: false,
    quotation: { found: false, match_type: "none" },
    interpretation: { provided: false, status: "error" },
    evidence: [],
  });
  assert.equal(view.isError, true);
  assert.equal(view.canonical, "");
  assert.equal(view.reference, "");
  assert.equal(view.dangerApplicable, false);
  assert.match(view.errorMessage, /تعذّر/);
});

test("inconclusive result keeps honest uncertainty", () => {
  const view = verificationView({
    verification_status: "inconclusive",
    status_label: "لا توجد أدلة كافية",
    status_explanation: "عدم العثور على دليل لا يعني أن النص غير صحيح.",
    input_text: "نص",
    text_status: "غير مؤكد",
    danger_level: "غير محسوبة",
    danger_level_applicable: false,
    quotation: { found: false, match_type: "none" },
    interpretation: { provided: false, status: "not_provided" },
    evidence: [],
  });
  assert.equal(view.isInconclusive, true);
  assert.equal(view.canonical, "");
  assert.equal(view.dangerApplicable, false);
  assert.match(view.explanation, /لا يعني/);
});

test("canonical evidence is not duplicated in supporting evidence", () => {
  const view = verificationView(PARTIAL);
  assert.equal(view.evidence.length, 2);
  const supporting = view.evidence.filter(
    (item) => item.role !== "canonical",
  );
  assert.equal(supporting.length, 1);
  assert.equal(supporting[0].reference, "تفسير ابن كثير 4:43");
});

test("missing or malformed data is safe", () => {
  assert.equal(verificationView(null), null);
  assert.equal(verificationView("text"), null);
});

const POST = {
  verification_status: "misleading",
  overall_status: "misleading",
  status_label: "تحذير — اقتباس أو استدلال مضلّل",
  status_explanation: "شرح غير مدعوم بالمصدر",
  analysis_mode: "post",
  component_count: 2,
  input_text: "منشور كامل",
  danger_level: "عالية",
  danger_level_applicable: true,
  danger_level_meaning: "احتمال الفهم المضلل، وليست درجة ثقة.",
  components: [
    {
      id: "c1",
      type: "quran_quote",
      type_label: "اقتباس قرآني",
      submitted_excerpt: "ولا تقربوا الصلاة",
      claim_text: "الصلاة محرمة على المسلمين",
      source_identity: { status: "partial", label: "جزء من آية مطابقة" },
      canonical_text: "يَا أَيُّهَا الَّذِينَ آمَنُوا...",
      reference: "النساء:43",
      completeness: {
        status: "partial_material",
        label: "اقتباس ناقص يحذف سياقاً مؤثراً",
        is_material: true,
        omitted_text: "وَأَنتُمْ سُكَارَىٰ...",
        explanation: "النقص يغير المعنى.",
      },
      authenticity: {
        status: "not_applicable",
        label: "نص قرآني ثابت في المصحف",
        assessment: "",
        source: "النساء:43",
      },
      interpretation: {
        status: "contradicted",
        label: "الشرح غير مدعوم بالمصدر",
        explanation: "الآية في حال السكر.",
        correction: "النهي متعلق بحال السكر.",
      },
      argument: {
        status: "misleading",
        label: "استدلال مضلّل",
        explanation: "الاستنتاج أوسع من الدليل.",
      },
      evidence: [
        { reference: "النساء:43", text: "الآية", role: "canonical" },
      ],
      severity: "high",
    },
    {
      id: "c2",
      type: "hadith_quote",
      type_label: "اقتباس حديثي",
      submitted_excerpt: "نص غير موجود",
      claim_text: "وهذا حديث صحيح.",
      source_identity: { status: "not_found", label: "لم يُعثر عليه في قاعدة البيانات المتاحة" },
      canonical_text: "",
      reference: "",
      completeness: {
        status: "unknown",
        label: "تعذّر تقييم الاكتمال",
        explanation: "لا يوجد مصدر مطابق.",
      },
      authenticity: {
        status: "not_found",
        label: "لم يُعثر عليه في قاعدة البيانات المتاحة",
        assessment: "عدم العثور لا يعني أنه موضوع.",
      },
      interpretation: {
        status: "inconclusive",
        label: "غير محسوم — الأدلة غير كافية",
        explanation: "لا توجد أدلة كافية.",
        correction: "",
      },
      argument: {
        status: "inconclusive",
        label: "لم يتم حسم الاستدلال",
        explanation: "لا توجد أدلة كافية.",
      },
      evidence: [],
      severity: "medium",
    },
  ],
};

test("post analysis exposes separate source, context, authenticity, interpretation and argument views", () => {
  const view = verificationView(POST);
  assert.equal(view.analysisMode, "post");
  assert.equal(view.overallStatus, "misleading");
  assert.equal(view.components.length, 2);

  const [quran, hadith] = view.components;
  assert.equal(quran.typeLabel, "اقتباس قرآني");
  assert.equal(quran.sourceLabel, "جزء من آية مطابقة");
  assert.equal(quran.completenessTone, "danger");
  assert.equal(quran.isMaterial, true);
  assert.equal(quran.interpretationTone, "danger");
  assert.equal(quran.argumentTone, "danger");
  assert.equal(quran.authenticityLabel, "نص قرآني ثابت في المصحف");

  assert.equal(hadith.sourceTone, "neutral");
  assert.equal(hadith.authenticityTone, "neutral");
  assert.equal(hadith.interpretationLabel, "غير محسوم — الأدلة غير كافية");
  assert.equal(hadith.argumentLabel, "لم يتم حسم الاستدلال");
});

test("post components never turn a not-found narration into a fabricated one", () => {
  const view = verificationView(POST);
  const hadith = view.components[1];
  assert.equal(hadith.authenticityStatus, "not_found");
  assert.match(hadith.authenticityAssessment, /لا يعني/);
  assert.doesNotMatch(hadith.authenticityLabel, /موضوعة/);
});

test("post analysis preserves the submitted text and canonical passage separately", () => {
  const view = verificationView(POST);
  assert.equal(view.submitted, "منشور كامل");
  assert.equal(view.components[0].excerpt, "ولا تقربوا الصلاة");
  assert.equal(view.components[0].canonical, POST.components[0].canonical_text);
  assert.equal(view.components[0].reference, "النساء:43");
});

test("evidence count only counts entries that can actually be rendered", () => {
  const data = {
    analysis_mode: "post",
    overall_status: "partially_verified",
    verification_status: "partially_verified",
    input_text: "post",
    components: [
      {
        id: "c1",
        type: "hadith_quote",
        type_label: "اقتباس حديثي",
        submitted_excerpt: "quote",
        source_identity: { status: "partial", label: "جزء من متن حديث مطابق" },
        completeness: {
          status: "partial_not_material",
          label: "اقتباس ناقص غير مؤثر",
          is_material: false,
        },
        authenticity: {
          status: "sahih_collection",
          label: "أخرجه صحيح البخاري",
        },
        interpretation: {
          status: "inconclusive",
          label: "غير محسوم — الأدلة غير كافية",
        },
        argument: {
          status: "unsupported",
          label: "الاستدلال غير مدعوم بالمصدر",
        },
        evidence: [
          {
            source: "hadith",
            reference: "صحيح البخاري:1",
            text: "إِنَّمَا الْأَعْمَالُ بِالنِّيَّاتِ",
            role: "canonical",
          },
          {
            source: "hadith",
            reference: "سنن أبي داود:2202",
            text: "",
            role: "additional_reference",
          },
        ],
        severity: "medium",
      },
    ],
  };

  const view = verificationView(data);
  const component = view.components[0];

  assert.equal(component.evidence.length, 1);
  assert.equal(component.evidence[0].reference, "صحيح البخاري:1");
  assert.equal(component.evidence[0].sourceLabel, "حديث");
  assert.ok(component.evidence[0].text);
  assert.match(component.interpretationHelp, /لم تتوفر أدلة كافية/);
  assert.match(component.argumentHelp, /صحة الاقتباس لا تعني/);
  assert.notEqual(component.interpretationHelp, component.argumentHelp);
});

test("unrelated retrieved evidence is labeled and not presented as support", () => {
  const view = verificationView({
    verification_status: "inconclusive",
    status_label: "لا توجد أدلة كافية",
    status_explanation: "لم يتم العثور على نص مرتبط.",
    input_text: "ادعاء طبي بلا نص.",
    text_status: "غير مؤكد",
    danger_level: "غير محسوبة",
    danger_level_applicable: false,
    quotation: { found: false, match_type: "none" },
    interpretation: { provided: false, status: "not_provided" },
    evidence: [],
    evidence_status: "not_relevant",
    evidence_note: "تم البحث في المصادر، لكنها لا تدعم الادعاء مباشرة.",
  });

  assert.equal(view.evidenceStatus, "not_relevant");
  assert.match(view.evidenceNote, /لا تدعم الادعاء مباشرة/);
  assert.equal(view.evidence.length, 0);
});
