import { useRef, useState } from "react";
import { getTafsir } from "../services/basiraService";
import FormattedResult from "../components/FormattedResult";
import { arabicMessage } from "../utils/format";
import "../styles/tools.css";

const RESULT_FALLBACK =
  "لم تصل تفاصيل التفسير. جرّب البحث مرة أخرى أو اكتب الآية بشكل أوضح.";

function Tafsir() {
  const [ayah, setAyah] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [fieldError, setFieldError] = useState("");

  const textareaRef = useRef(null);

  async function runSearch() {
    if (loading) return;

    const cleanAyah = ayah.trim();

    if (!cleanAyah) {
      setFieldError("اكتب الآية أو جزءًا منها للبحث.");
      textareaRef.current?.focus();
      return;
    }

    setFieldError("");
    setError("");
    setResult(null);
    setLoading(true);

    try {
      const response = await getTafsir(cleanAyah);

      if (!response?.data) {
        setError(RESULT_FALLBACK);
        return;
      }

      setResult(response.data);
    } catch (requestError) {
      setError(
        arabicMessage(
          requestError.response?.data?.message,
          "تعذّر جلب التفسير الآن. راجع اتصالك ثم حاول مرة أخرى."
        )
      );
    } finally {
      setLoading(false);
    }
  }

  function handleSubmit(event) {
    event.preventDefault();
    runSearch();
  }

  function handleKeyDown(event) {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      runSearch();
    }
  }

  return (
    <main className="page tool-page">
      <div className="container tool-shell">
        <header className="page-header tool-header" data-reveal>
          <p className="eyebrow">أداة التفسير</p>
          <h1>ابحث عن آية، واقرأ تفسيرها بهدوء</h1>
          <p>
            اكتب الآية كما تتذكرها، أو جزءًا واضحًا منها. ستعرض الأداة
            التفسير والسياق والمصدر في صفحة واحدة مرتبة.
          </p>
        </header>

        <div className="tool-grid">
          <div className="tool-main">
            <section className="tool-card" aria-labelledby="tafsir-form-title">
              <div className="tool-card-head">
                <span className="service-icon" aria-hidden="true">
                  <i className="bi bi-book"></i>
                </span>
                <div>
                  <h2 id="tafsir-form-title">الآية المطلوبة</h2>
                  <p>كلما كانت الآية أدق، كان الوصول إلى التفسير أسهل.</p>
                </div>
              </div>

              <form className="tool-form" onSubmit={handleSubmit} noValidate>
                <div className="field">
                  <label htmlFor="tafsir-ayah">
                    الآية أو جزء منها
                    <span className="tool-required">مطلوب</span>
                  </label>
                  <textarea
                    id="tafsir-ayah"
                    ref={textareaRef}
                    value={ayah}
                    onChange={function (event) {
                      setAyah(event.target.value);
                      if (fieldError) setFieldError("");
                      if (error) setError("");
                    }}
                    onKeyDown={handleKeyDown}
                    placeholder="اكتب الآية هنا…"
                    rows={5}
                    aria-invalid={Boolean(fieldError)}
                    aria-describedby={
                      fieldError ? "tafsir-ayah-error" : "tafsir-ayah-hint"
                    }
                    disabled={loading}
                  />
                  {fieldError ? (
                    <span
                      className="field-error"
                      id="tafsir-ayah-error"
                      role="alert"
                    >
                      <i
                        className="bi bi-exclamation-circle"
                        aria-hidden="true"
                      ></i>
                      {fieldError}
                    </span>
                  ) : (
                    <span className="field-hint" id="tafsir-ayah-hint">
                      يمكنك كتابة الآية كاملة، أو مقطع قصير منها يميزها.
                    </span>
                  )}
                </div>

                <div className="tool-form-foot">
                  <span className="field-hint">
                    <i className="bi bi-journal-text" aria-hidden="true"></i>
                    النتيجة تشمل التفسير والمصدر عند توفره.
                  </span>

                  <button
                    className="btn btn-primary tool-submit"
                    type="submit"
                    disabled={loading || !ayah.trim()}
                    aria-busy={loading}
                  >
                    {loading ? (
                      <>
                        <span className="spinner" aria-hidden="true"></span>
                        جارٍ البحث…
                      </>
                    ) : (
                      <>
                        ابحث عن التفسير
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
                  onClick={runSearch}
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
                    <strong>جارٍ البحث في التفسير…</strong>
                    <p>نحدد الآية ثم نراجع تفسيرها في المصادر.</p>
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
                aria-labelledby="tafsir-result-title"
              >
                <header className="tool-result-head">
                  <span
                    className="tool-result-head-icon"
                    aria-hidden="true"
                  >
                    <i className="bi bi-book-half"></i>
                  </span>
                  <div>
                    <h2 id="tafsir-result-title">نتيجة التفسير</h2>
                    <p>
                      اقرأ النص كاملًا، وارجع إلى المصدر إن أردت المزيد من
                      السياق.
                    </p>
                  </div>
                </header>

                <FormattedResult data={result} />
              </section>
            ) : null}
          </div>

          <aside className="tool-guide" aria-labelledby="tafsir-guide-title">
            <h2 id="tafsir-guide-title">للحصول على نتيجة أدق</h2>
            <ol className="tool-guide-list">
              <li>
                <span>
                  <strong>اكتب الآية بدقة</strong>
                  استخدم الكلمات كما تتذكرها، مع علامات التشكيل إن أمكن.
                </span>
              </li>
              <li>
                <span>
                  <strong>أضف ما يميز الآية</strong>
                  إذا كانت الكلمات قصيرة أو متكررة، أضف الجزء الذي يليها.
                </span>
              </li>
              <li>
                <span>
                  <strong>راجع المصدر</strong>
                  التفسير مساعدة للفهم، ويبقى الرجوع إلى المصدر خطوة مهمة.
                </span>
              </li>
            </ol>

            <div className="tool-guide-note">
              <i className="bi bi-info-circle" aria-hidden="true"></i>
              <span>
                تعتمد النتيجة على المصادر المتاحة في بصيرة، وقد لا تشمل كل
                ما ورد في كتب التفسير.
              </span>
            </div>
          </aside>
        </div>
      </div>
    </main>
  );
}

export default Tafsir;
