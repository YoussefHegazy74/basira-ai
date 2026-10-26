const MATCH_LABELS = {
  exact: "اقتباس مطابق",
  partial: "اقتباس جزئي",
  approximate: "اقتباس تقريبي",
  semantic: "مدعوم في المصادر",
  ambiguous: "تطابق غير حاسم",
  none: "لم يتم العثور على اقتباس",
};

const STATUS_LABELS = {
  verified: "تم التحقق",
  partial: "تم التحقق — اقتباس جزئي",
  contradicted: "المعلومة تخالف الأدلة",
  inconclusive: "لا توجد أدلة كافية",
  error: "تعذّر إكمال التحقق",
};

const TONES = {
  verified: "success",
  partial: "warning",
  contradicted: "danger",
  misleading: "danger",
  partially_verified: "warning",
  inconclusive: "warning",
  error: "danger",
};

const INTERPRETATION_LABELS = {
  supported: "التفسير مدعوم بالمصادر",
  contradicted: "التفسير غير مدعوم أو يخالف المصادر",
  partially_supported: "التفسير مدعوم جزئياً",
  inconclusive: "المصادر لا تكفي للحكم على التفسير",
  error: "تعذّر تقييم التفسير",
  not_provided: "لم يتم تقديم تفسير لتقييمه",
};

const ARGUMENT_LABELS = {
  supported: "الاستدلال مدعوم",
  unsupported: "الاستدلال غير مدعوم بالمصدر",
  overstated: "الاستدلال أقوى مما يثبته المصدر",
  misleading: "استدلال مضلّل",
  disputed: "استدلال محل خلاف",
  inconclusive: "لم يتم حسم الاستدلال",
  not_assessed: "لم يتم تقديم استدلال صريح لتقييمه",
};

const AUTHENTICITY_TONES = {
  sahih_collection: "success",
  hasan_sahih: "success",
  sahih: "success",
  hasan: "success",
  daif: "warning",
  disputed: "warning",
  not_recorded: "warning",
  mawdu: "danger",
  munkar: "danger",
  not_authentic: "danger",
  batil: "danger",
  not_found: "neutral",
  not_applicable: "neutral",
};

const COMPLETENESS_TONES = {
  complete: "success",
  partial_not_material: "warning",
  partial_material: "danger",
  altered: "danger",
  unknown: "neutral",
};

const INTERPRETATION_TONES = {
  supported: "success",
  contradicted: "danger",
  partially_supported: "warning",
  disputed: "warning",
  inconclusive: "warning",
  not_provided: "neutral",
  error: "danger",
};

const ARGUMENT_TONES = {
  supported: "success",
  unsupported: "danger",
  misleading: "danger",
  overstated: "warning",
  disputed: "warning",
  inconclusive: "warning",
  not_assessed: "neutral",
};

const SOURCE_TYPE_LABELS = {
  quran: "القرآن",
  hadith: "حديث",
  tafsir: "تفسير",
  fiqh: "فقه",
  sira: "سيرة",
  grading: "تقييم",
};

const INTERPRETATION_HELP = {
  supported: "المصادر المتاحة تدعم هذا الشرح.",
  contradicted: "المصادر المتاحة لا تدعم هذا الشرح.",
  partially_supported: "المصادر تدعم جزءاً من الشرح لا كله.",
  disputed: "الأدلة المتاحة تسجل اختلافاً تفسيرياً؛ راجعها قبل الجزم.",
  inconclusive: "لم تتوفر أدلة كافية لتقييم الشرح نفسه. هذا لا يعني أن الشرح صحيح أو خاطئ.",
  not_provided: "لم يتم تقديم تفسير صريح لتقييمه.",
  error: "تعذّر تقييم الشرح.",
};

const ARGUMENT_HELP = {
  supported: "المصدر يدعم النتيجة التي ذكرها المؤلف.",
  unsupported: "المصدر لا يذكر النتيجة التي استخلصها المؤلف؛ صحة الاقتباس لا تعني أن الاستنتاج مدعوم.",
  overstated: "النتيجة أقوى مما يثبته المصدر.",
  misleading: "النتيجة المستخلصة تتجاوز أو تحرف ما يثبته المصدر.",
  disputed: "الأدلة المتاحة تسجل اختلافاً في الاستدلال؛ راجعها قبل الجزم.",
  inconclusive: "الأدلة لا تكفي للحكم على الاستدلال.",
  not_assessed: "لم يتم تقديم استدلال صريح لتقييمه.",
};

function clean(value) {
  return typeof value === "string" ? value.trim() : "";
}

function componentView(component) {
  if (!component || typeof component !== "object") return null;
  const source = component.source_identity || {};
  const completeness = component.completeness || {};
  const authenticity = component.authenticity || {};
  const interpretation = component.interpretation || {};
  const argument = component.argument || {};
  const sourceStatus = clean(source.status) || "inconclusive";
  const completenessStatus = clean(completeness.status) || "unknown";
  const authenticityStatus = clean(authenticity.status) || "not_applicable";
  const interpretationStatus =
    clean(interpretation.status) || "not_provided";
  const argumentStatus = clean(argument.status) || "not_assessed";
  const severity = clean(component.severity) || "none";

  return {
    id: component.id,
    type: clean(component.type),
    typeLabel: clean(component.type_label) || "عنصر",
    excerpt: clean(component.submitted_excerpt),
    claim: clean(component.claim_text),
    sourceStatus,
    sourceLabel: clean(source.label) || "حالة المصدر",
    sourceTone:
      sourceStatus === "verified"
        ? "success"
        : sourceStatus === "partial" || sourceStatus === "approximate"
          ? "warning"
          : "neutral",
    canonical: clean(component.canonical_text),
    canonicalLabel:
      clean(component.canonical_label) || "النص الكامل من المصدر",
    reference: clean(component.reference),
    completenessStatus,
    completenessLabel: clean(completeness.label) || "حالة الاكتمال",
    completenessTone: COMPLETENESS_TONES[completenessStatus] || "neutral",
    omittedText: clean(completeness.omitted_text),
    completenessExplanation: clean(completeness.explanation),
    materialityBasis: clean(completeness.materiality_basis),
    isMaterial: Boolean(completeness.is_material),
    authenticityStatus,
    authenticityLabel: clean(authenticity.label),
    authenticityTone:
      AUTHENTICITY_TONES[authenticityStatus] || "neutral",
    authenticityAssessment: clean(authenticity.assessment),
    assessor: clean(authenticity.assessor),
    authenticitySource: clean(authenticity.source),
    interpretationStatus,
    interpretationLabel:
      clean(interpretation.label) ||
      INTERPRETATION_LABELS[interpretationStatus] ||
      "حالة التفسير",
    interpretationTone:
      INTERPRETATION_TONES[interpretationStatus] || "neutral",
    interpretationExplanation: clean(interpretation.explanation),
    correction: clean(interpretation.correction),
    argumentStatus,
    argumentLabel:
      clean(argument.label) || ARGUMENT_LABELS[argumentStatus] || "حالة الاستدلال",
    argumentTone: ARGUMENT_TONES[argumentStatus] || "neutral",
    argumentExplanation: clean(argument.explanation),
    interpretationHelp:
      INTERPRETATION_HELP[interpretationStatus] || "",
    argumentHelp: ARGUMENT_HELP[argumentStatus] || "",
    missingInformation: Array.isArray(component.missing_information)
      ? component.missing_information.filter((item) => clean(item))
      : [],
    evidence: Array.isArray(component.evidence)
      ? component.evidence
          .filter(
            (item) =>
              item &&
              typeof item === "object" &&
              clean(item.text),
          )
          .map((item) => ({
            ...item,
            sourceLabel:
              SOURCE_TYPE_LABELS[clean(item.source)] ||
              clean(item.source) ||
              "مصدر",
          }))
      : [],
    severity,
    severityTone:
      severity === "high"
        ? "danger"
        : severity === "medium"
          ? "warning"
          : severity === "low"
            ? "success"
            : "neutral",
  };
}

export function verificationView(data) {
  if (!data || typeof data !== "object") return null;

  const status = clean(data.verification_status) || "inconclusive";
  const quotation =
    data.quotation && typeof data.quotation === "object"
      ? data.quotation
      : {};
  const interpretation =
    data.interpretation && typeof data.interpretation === "object"
      ? data.interpretation
      : {};

  const submitted = clean(data.input_text);
  const canonical = clean(quotation.canonical_text || data.full_text);
  const reference = clean(quotation.reference || data.source);
  const matchType = clean(quotation.match_type) || "none";
  const interpretationProvided = Boolean(
    interpretation.provided || clean(data.input_explanation)
  );
  const interpretationStatus = clean(interpretation.status) ||
    (interpretationProvided ? "inconclusive" : "not_provided");

  const errorMessage = clean(data.verification_error);
  const isError = status === "error";

  return {
    status,
    tone: TONES[status] || "warning",
    label: clean(data.status_label) || STATUS_LABELS[status] || "نتيجة التحقق",
    explanation: clean(data.status_explanation),
    submitted,
    submissionLabel: "النص المُدخل",
    canonical:
      !isError && quotation.found && canonical ? canonical : "",
    canonicalLabel:
      matchType === "semantic"
        ? "النص الكامل من المصدر"
        : "النص الكامل من المصدر",
    reference: !isError ? reference : "",
    sourceLabel: "المصدر والمرجع",
    matchType,
    matchLabel: MATCH_LABELS[matchType] || "",
    quotationFound: Boolean(quotation.found),
    isPartial: Boolean(quotation.is_partial),
    isComplete: Boolean(quotation.is_complete),
    completenessLabel: quotation.found
      ? quotation.is_complete
        ? "الاقتباس كامل"
        : "الاقتباس جزئي"
      : "",
    interpretationProvided,
    interpretationStatus,
    interpretationLabel:
      clean(interpretation.label) ||
      INTERPRETATION_LABELS[interpretationStatus] ||
      "لم يتم تقييم التفسير",
    interpretationTone:
      interpretationStatus === "supported"
        ? "success"
        : interpretationStatus === "contradicted"
          ? "danger"
          : interpretationStatus === "partially_supported"
            ? "warning"
            : interpretationStatus === "error"
              ? "danger"
              : "neutral",
    interpretationAssessment: clean(interpretation.assessment),
    interpretationCorrection: clean(interpretation.correction),
    interpretationError: clean(interpretation.error),
    evidence: Array.isArray(data.evidence)
      ? data.evidence.filter(
          (item) =>
            item &&
            typeof item === "object" &&
            (clean(item.text) || clean(item.reference))
        )
      : [],
    evidenceStatus:
      clean(data.evidence_status) ||
      (Array.isArray(data.evidence) && data.evidence.length
        ? "relevant"
        : "none"),
    evidenceNote: clean(data.evidence_note),
    dangerLevel: clean(data.danger_level),
    dangerApplicable: Boolean(data.danger_level_applicable),
    dangerMeaning: clean(data.danger_level_meaning),
    errorMessage: errorMessage || (isError ? clean(data.status_explanation) : ""),
    isError,
    isInconclusive: status === "inconclusive",
    analysisMode: clean(data.analysis_mode),
    overallStatus: clean(data.overall_status) || status,
    componentCount: Number(data.component_count) || 0,
    components: Array.isArray(data.components)
      ? data.components.map(componentView).filter(Boolean)
      : [],
  };
}

export { MATCH_LABELS, STATUS_LABELS };
