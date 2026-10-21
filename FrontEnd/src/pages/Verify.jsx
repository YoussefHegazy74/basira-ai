
import { useState } from "react";
import { verifyText } from "../services/basiraService";
import FormattedResult from "../components/FormattedResult";

function Verify() {
  const [text, setText] = useState("");
  const [explanation, setExplanation] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(event) {
    event.preventDefault();

    if (!text.trim() || loading) {
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await verifyText(
        text.trim(),
        explanation.trim() || undefined
      );

      setResult(response.data);
    } catch (err) {
      setError(
        err.response?.data?.message ||
          "حدث خطأ أثناء التحقق من النص. حاول مرة أخرى."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="service-page">
      <div className="container service-page-content">
        <span className="section-label">
          Information Verification
        </span>

        <h1>التحقق من المعلومات</h1>

        <p className="page-description">
          أدخل النص الذي تريد فحصه للحصول على نتيجة التحقق.
        </p>

        <section className="tool-panel">
          <div className="tool-heading">
            <span className="service-icon">
              <i className="bi bi-shield-check"></i>
            </span>

            <div>
              <h2>فحص النص</h2>
              <p>
                اكتب النص كاملًا للحصول على نتيجة أوضح.
              </p>
            </div>
          </div>

          <form onSubmit={handleSubmit}>
            <label htmlFor="verify-text">
              النص المراد التحقق منه
            </label>

            <textarea
              id="verify-text"
              value={text}
              onChange={function (event) {
                setText(event.target.value);
              }}
              placeholder="اكتب أو الصق النص هنا..."
              rows={7}
              required
            />

            <label htmlFor="verify-explanation">
              توضيح إضافي (اختياري)
            </label>

            <textarea
              id="verify-explanation"
              value={explanation}
              onChange={function (event) {
                setExplanation(event.target.value);
              }}
              placeholder="أضف مصدر النص أو أي سياق يساعد في التحقق..."
              rows={3}
            />

            <button
              className="primary-button"
              type="submit"
              disabled={loading}
            >
              {loading ? "جارٍ التحقق..." : "تحقق من النص"}
              {" "}
              <i className="bi bi-search"></i>
            </button>
          </form>

          {error && (
            <p role="alert" className="api-error">
              {error}
            </p>
          )}

          {result !== null && (
            <section
              className="api-result"
              aria-live="polite"
            >
              <h2>نتيجة التحقق</h2>
              <FormattedResult data={result} />
            </section>
          )}
        </section>
      </div>
    </main>
  );
}

export default Verify;
