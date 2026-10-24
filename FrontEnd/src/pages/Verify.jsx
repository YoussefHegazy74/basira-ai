import { useRef, useState } from "react";
import { verifyText } from "../services/basiraService";
import FormattedResult from "../components/FormattedResult";
import { arabicMessage } from "../utils/format";
import "../styles/tools.css";

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
          <h1>تحقق من النص قبل أن تعتمد عليه</h1>
          <p>
            الصق نصًا أو معلومة لديك. ستعرض الأداة ما تدعمه المصادر وما يحتاج
            إلى مراجعة، لتصل إلى قرار أوضح.
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
                  <h2 id="verify-form-title">النص المراد التحقق منه</h2>
                  <p>أدخل النص كما هو، ثم راجع النتيجة بهدوء.</p>
                </div>
              </div>

              <form className="tool-form" onSubmit={handleSubmit} noValidate>
                <div className="field">
                  <label htmlFor="verify-text">
                    النص
                    <span className="tool-required">مطلوب</span>
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
                    placeholder="اكتب أو الصق النص الذي تريد فحصه…"
                    rows={7}
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
                      كلما كان النص كاملًا وواضحًا، كانت المراجعة أدق.
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
                    <strong>جارٍ مراجعة النص…</strong>
                    <p>نقارن ما كتبته بما ورد في المصادر.</p>
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
                    <i className="bi bi-check2-circle"></i>
                  </span>
                  <div>
                    <h2 id="verify-result-title">نتيجة التحقق</h2>
                    <p>
                      اقرأ التفاصيل، وتذكّر أن مراجعة المصدر تبقى خطوة
                      أساسية.
                    </p>
                  </div>
                </header>

                <FormattedResult data={result} />
              </section>
            ) : null}
          </div>

          <aside className="tool-guide" aria-labelledby="verify-guide-title">
            <h2 id="verify-guide-title">قبل أن تبدأ</h2>
            <ol className="tool-guide-list">
              <li>
                <span>
                  <strong>اكتب النص كما هو</strong>
                  لا تعِد صياغته، حتى ترى النتيجة قراءة دقيقة لما لديك.
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
