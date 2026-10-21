
const labels = {
  answer: "الإجابة",
  tafsir: "التفسير",
  explanation: "الشرح",
  meaning: "المعنى",
  verse: "الآية",
  ayah: "الآية",
  reference: "المرجع",
  source: "المصدر",
  sources: "المصادر",
  conclusion: "الخلاصة",
  result: "النتيجة",
  status: "الحالة",
  confidence: "درجة الثقة",
  reason: "السبب",
  evidence: "الأدلة",
  details: "التفاصيل",
  verification: "نتيجة التحقق",
  is_valid: "صحة المعلومة",
  is_correct: "صحة المعلومة",
  summary: "الملخص",
};

function formatLabel(key) {
  if (labels[key]) {
    return labels[key];
  }

  return key
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/[_-]/g, " ")
    .replace(/\b\w/g, function (char) {
      return char.toUpperCase();
    });
}

function FormattedValue({ value }) {
  if (value === null || value === undefined || value === "") {
    return null;
  }

  if (typeof value === "boolean") {
    return <p>{value ? "نعم" : "لا"}</p>;
  }

  if (typeof value === "string" || typeof value === "number") {
    return <p className="formatted-result-text">{String(value)}</p>;
  }

  if (Array.isArray(value)) {
    if (value.length === 0) {
      return null;
    }

    return (
      <ul className="formatted-result-list">
        {value.map(function (item, index) {
          return (
            <li key={index}>
              {typeof item === "object" && item !== null ? (
                <FormattedValue value={item} />
              ) : (
                String(item)
              )}
            </li>
          );
        })}
      </ul>
    );
  }

  if (typeof value === "object") {
    return (
      <div className="formatted-result-fields">
        {Object.entries(value).map(function ([key, item]) {
          if (
            item === null ||
            item === undefined ||
            item === "" ||
            key === "__v"
          ) {
            return null;
          }

          return (
            <div className="formatted-result-field" key={key}>
              <h3>{formatLabel(key)}</h3>
              <FormattedValue value={item} />
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

  return (
    <div className="formatted-result" dir="rtl">
      <FormattedValue value={result} />
    </div>
  );
}

export default FormattedResult;
