const labels = {
  answer: "الإجابة",
  tafsir: "التفسير",
  explanation: "الشرح",
  meaning: "المعنى",
  verse: "الآية",
  ayah: "الآية",
  reference: "المرجع",
  source: "المصدر",
  source_scope: "نطاق نص المصدر",
  sources: "المصادر",
  conclusion: "الخلاصة",
  result: "النتيجة",
  status: "الحالة",
  confidence: "درجة الثقة",
  reason: "السبب",
  evidence: "الأدلة",
  details: "التفاصيل",
  verification: "نتيجة التحقق",
  verification_result: "نتيجة التحقق",
  is_valid: "صحة المعلومة",
  is_correct: "صحة المعلومة",
  summary: "الملخص",
  text_status: "حالة النص",
  is_truncated: "اكتمال النص",
  missing_part: "الجزء الناقص",
  full_text: "النص الكامل",
  danger_level: "درجة الخطورة",
  explanation_status: "تقييم الشرح",
  explanation_verdict: "سبب التقييم",
  correct_explanation: "الشرح الصحيح",
  input_text: "النص المُدخل",
  input_explanation: "الشرح المُدخل",
  context: "السياق",
  input_ayah: "الآية المُدخلة",
  title: "العنوان",
  text: "النص",
  url: "الرابط",
  page: "الصفحة",
  name: "الاسم",
  type: "النوع",
  date: "التاريخ",
  query: "الاستعلام",
};

const HIDDEN_KEYS = new Set([
  "__v",
  "_id",
  "id",
  "input_text",
  "input_explanation",
  "input_ayah",
]);

const LONG_TEXT_KEYS = new Set([
  "answer",
  "tafsir",
  "context",
  "full_text",
  "correct_explanation",
  "explanation_verdict",
  "explanation",
  "summary",
  "text",
]);

const FIELD_ORDER = [
  "text_status",
  "danger_level",
  "is_truncated",
  "missing_part",
  "full_text",
  "explanation_status",
  "explanation_verdict",
  "correct_explanation",
  "tafsir",
  "context",
  "answer",
  "summary",
  "conclusion",
  "sources",
  "source",
  "reference",
];

function labelFor(key) {
  return labels[key] || "تفصيل إضافي";
}

function shouldHide(item) {
  return item === null || item === undefined || item === "";
}

function fieldWeight(key) {
  const index = FIELD_ORDER.indexOf(key);
  return index === -1 ? FIELD_ORDER.length : index;
}

function statusTone(key, value) {
  if (typeof value !== "string") return "";

  if (key === "danger_level") {
    if (value.includes("عالية")) return "danger";
    if (value.includes("متوسطة")) return "warning";
    if (value.includes("منخفضة")) return "success";
  }

  if (key === "text_status") {
    if (value.includes("موضوع")) return "danger";
    if (value.includes("ضعيف")) return "warning";
    if (value.includes("صحيح")) return "success";
  }

  if (key === "explanation_status") {
    if (value.includes("مخالف")) return "danger";
    if (value.includes("جزئ")) return "warning";
    if (value.includes("موافق")) return "success";
  }

  return "";
}

const STATUS_KEYS = new Set([
  "text_status",
  "danger_level",
  "explanation_status",
  "status",
]);

function BooleanValue({ fieldKey, value }) {
  if (fieldKey === "is_truncated") {
    return (
      <p className="result-text">
        {value ? "نعم، النص مختصر أو ناقص." : "لا، النص مكتمل بحسب المصادر."}
      </p>
    );
  }

  return <p className="result-text">{value ? "نعم" : "لا"}</p>;
}

function FormattedValue({ value, fieldKey }) {
  if (shouldHide(value)) {
    return null;
  }

  if (typeof value === "boolean") {
    return <BooleanValue fieldKey={fieldKey} value={value} />;
  }

  if (typeof value === "string" || typeof value === "number") {
    if (STATUS_KEYS.has(fieldKey) && typeof value === "string") {
      return (
        <span className={`result-badge ${statusTone(fieldKey, value)}`}>
          {value}
        </span>
      );
    }

    return <p className="result-text">{String(value)}</p>;
  }

  if (Array.isArray(value)) {
    if (value.length === 0) {
      return null;
    }

    return (
      <ul className="result-list">
        {value.map(function (item, index) {
          const content =
            typeof item === "object" && item !== null ? (
              <FormattedValue value={item} />
            ) : (
              <span className="result-text">{String(item)}</span>
            );

          return <li key={index}>{content}</li>;
        })}
      </ul>
    );
  }

  if (typeof value === "object") {
    const entries = Object.entries(value).filter(function ([key, item]) {
      return !shouldHide(item) && !HIDDEN_KEYS.has(key);
    });

    if (!entries.length) {
      return null;
    }

    return (
      <div className="result-nested-grid">
        {entries.map(function ([key, item]) {
          return (
            <div className="result-nested-item" key={key}>
              <h4>{labelFor(key)}</h4>
              <FormattedValue value={item} fieldKey={key} />
            </div>
          );
        })}
      </div>
    );
  }

  return null;
}

function FormattedResult({ data }) {
  if (data === null || data === undefined) {
    return null;
  }

  // إخفاء غلاف الاستجابة العام إن كان موجودًا.
  const result =
    typeof data === "object" &&
    !Array.isArray(data) &&
    Object.prototype.hasOwnProperty.call(data, "data")
      ? data.data
      : data;

  if (shouldHide(result)) {
    return null;
  }

  if (typeof result === "string" || typeof result === "number") {
    return (
      <div className="formatted-result" dir="rtl">
        <p className="result-text">{String(result)}</p>
      </div>
    );
  }

  if (typeof result !== "object" || Array.isArray(result)) {
    return (
      <div className="formatted-result" dir="rtl">
        <FormattedValue value={result} />
      </div>
    );
  }

  const entries = Object.entries(result)
    .filter(function ([key, item]) {
      return !shouldHide(item) && !HIDDEN_KEYS.has(key);
    })
    .sort(function ([firstKey], [secondKey]) {
      return fieldWeight(firstKey) - fieldWeight(secondKey);
    });

  if (!entries.length) {
    return (
      <p className="result-text muted">
        وصلت الاستجابة دون تفاصيل إضافية. جرّب صياغة أخرى أو أعد المحاولة.
      </p>
    );
  }

  return (
    <div className="formatted-result" dir="rtl">
      <div className="result-grid">
        {entries.map(function ([key, item]) {
          const isWide = LONG_TEXT_KEYS.has(key);

          return (
            <div
              className={`result-field ${isWide ? "result-field-wide" : ""}`}
              key={key}
            >
              <h3>{labelFor(key)}</h3>
              <FormattedValue value={item} fieldKey={key} />
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default FormattedResult;
