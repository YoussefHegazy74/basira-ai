import { useRef, useState } from "react";
import { verifyText } from "../services/basiraService";
import VerificationResult from "../components/VerificationResult";
import { arabicMessage } from "../utils/format";
import "../styles/tools.css";
import "../styles/verification.css";

const RESULT_FALLBACK =
  "لم تصل تفاصيل النتيجة. جرّب مرة أخرى أو اختصر النص قليلًا.";

function Verify() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [fieldError, setFieldError] = useState("");

  const textareaRef = useRef(null);

  async function runVerification() {
    if (loading) return;

    const cleanText = text.trim();

    if (!cleanText) {
      setFieldError("اكتب النص الذي تريد التحقق منه أولًا.");
      textareaRef.current?.focus();
      return;
    }

    setFieldError("");
    setError("");
    setResult(null);
    setLoading(true);

    try {
      const response = await verifyText(cleanText);

      if (!response?.data) {
        setError(RESULT_FALLBACK);
        return;
      }

      setResult(response.data);
    } catch (requestError) {
      setError(
        arabicMessage(
          requestError.response?.data?.message,
          "تعذّر إكمال التحقق الآن. راجع اتصالك ثم حاول مرة أخرى."
        )
      );
    } finally {
      setLoading(false);
    }
  }

  function handleSubmit(event) {
    event.preventDefault();
    runVerification();
  }

  function handleKeyDown(event) {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      runVerification();
    }
  }

  return (
    <main className="page tool-page">
      <div className="container tool-shell">
        <header className="page-header tool-header" data-reveal>
          <p className="eyebrow">أداة التحقق</p>
          <h1>تحقق من المنشور كاملًا قبل أن تعتمد عليه</h1>
          <p>
            الصق المنشور كما نُشر: الاقتباسات، والشرح، والنتيجة التي يريد
            الكاتب الوصول إليها. ستفصل الأداة بين صحة النقل، واكتمال السياق،
            وصحة الاستدلال.
          </p>
        </header>

        <div className="tool-grid">
          <div className="tool-main">
            <section className="tool-card" aria-labelledby="verify-form-title">
              <div className="tool-card-head">
                <span className="service-icon" aria-hidden="true">
                  <i className="bi bi-shield-check"></i>
                </span>
                <div>
                  <h2 id="verify-form-title">المنشور المراد التحقق منه</h2>
                  <p>أدخل المنشور كاملًا كما هو، ثم راجع النتيجة بهدوء.</p>
                </div>
              </div>

              <form className="tool-form" onSubmit={handleSubmit} noValidate>
                <div className="field">
                  <label htmlFor="verify-text">
                    <span>
                      المنشور كاملًا{" "}
                      <span className="tool-required">مطلوب</span>
                    </span>
                  </label>
                  <textarea
                    id="verify-text"
                    ref={textareaRef}
                    className="tool-textarea-lg"
                    value={text}
                    onChange={function (event) {
                      setText(event.target.value);
                      if (fieldError) setFieldError("");
                      if (error) setError("");
                    }}
                    onKeyDown={handleKeyDown}
                    placeholder="الصق المنشور هنا: الاقتباسات والشرح والاستنتاجات…"
                    rows={10}
                    aria-invalid={Boolean(fieldError)}
                    aria-describedby={
                      fieldError ? "verify-text-error" : "verify-text-hint"
                    }
                    disabled={loading}
                  />
                  {fieldError ? (
                    <span
                      className="field-error"
                      id="verify-text-error"
                      role="alert"
                    >
                      <i
                        className="bi bi-exclamation-circle"
                        aria-hidden="true"
                      ></i>
                      {fieldError}
                    </span>
                  ) : (
                    <span className="field-hint" id="verify-text-hint">
                      كلما كان المنشور كاملًا، أمكن فصل الاقتباس عن الشرح
                      والاستنتاج بدقة.
                    </span>
                  )}
                </div>

                <div className="tool-form-foot">
                  <span className="field-hint">
                    <i className="bi bi-clock" aria-hidden="true"></i>
                    قد تستغرق المراجعة لحظات.
                  </span>

                  <button
                    className="btn btn-primary tool-submit"
                    type="submit"
                    disabled={loading || !text.trim()}
                    aria-busy={loading}
                  >
                    {loading ? (
                      <>
                        <span className="spinner" aria-hidden="true"></span>
                        جارٍ التحقق…
                      </>
                    ) : (
                      <>
                        ابدأ التحقق
                        <i
                          className="bi bi-arrow-left"
                          aria-hidden="true"
                        ></i>
                      </>
                    )}
                  </button>
                </div>
              </form>
            </section>

            {error ? (
              <div className="alert alert-error" role="alert">
                <i
                  className="bi bi-exclamation-circle"
                  aria-hidden="true"
                ></i>
                <span>{error}</span>
                <button
                  type="button"
                  className="btn btn-secondary btn-sm alert-action"
                  onClick={runVerification}
                  disabled={loading}
                >
                  إعادة المحاولة
                </button>
              </div>
            ) : null}

            {loading ? (
              <section
                className="tool-loading-card"
                aria-live="polite"
                aria-busy="true"
              >
                <div className="tool-loading-head">
                  <span className="spinner" aria-hidden="true"></span>
                  <div>
                    <strong>جارٍ تحليل المنشور…</strong>
                    <p>نفصل الاقتباسات عن الشرح والاستنتاج ونراجعها.</p>
                  </div>
                </div>
                <div className="tool-skeleton-lines" aria-hidden="true">
                  <span className="skeleton"></span>
                  <span className="skeleton"></span>
                  <span className="skeleton"></span>
                  <span className="skeleton"></span>
                </div>
              </section>
            ) : null}

            {!loading && result !== null ? (
              <section
                className="tool-result"
                aria-labelledby="verify-result-title"
              >
                <header className="tool-result-head">
                  <span
                    className="tool-result-head-icon"
                    aria-hidden="true"
                  >
                    <i className="bi bi-clipboard-check"></i>
                  </span>
                  <div>
                    <h2 id="verify-result-title">نتيجة التحقق</h2>
                    <p>
                      اقرأ التفاصيل، وتذكّر أن مراجعة المصدر تبقى خطوة
                      أساسية.
                    </p>
                  </div>
                </header>

                <VerificationResult data={result} />
              </section>
            ) : null}
          </div>

          <aside className="tool-guide" aria-labelledby="verify-guide-title">
            <h2 id="verify-guide-title">قبل أن تبدأ</h2>
            <ol className="tool-guide-list">
              <li>
                <span>
                  <strong>الصق المنشور كاملًا</strong>
                  لا تعِد صياغته، حتى ترى النتيجة قراءة دقيقة للاقتباس
                  والشرح والاستنتاج.
                </span>
              </li>
              <li>
                <span>
                  <strong>افصل بين ثلاثة أسئلة</strong>
                  هل النقل صحيح؟ هل السياق مكتمل؟ هل الاستدلال يدعمه
                  المصدر؟
                </span>
              </li>
              <li>
                <span>
                  <strong>راجع المصدر بنفسك</strong>
                  النتيجة مساعدة للفهم، وليست حكمًا نهائيًا على المعلومة.
                </span>
              </li>
            </ol>

            <div className="tool-guide-note">
              <i className="bi bi-info-circle" aria-hidden="true"></i>
              <span>
                في المسائل الدينية أو الطبية أو القانونية، ارجع إلى أهل
                الاختصاص قبل الاعتماد على أي نتيجة.
              </span>
            </div>
          </aside>
        </div>
      </div>
    </main>
  );
}

export default Verify;
