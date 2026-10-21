
import { useState } from "react";
import { getTafsir } from "../services/basiraService";
import FormattedResult from "../components/FormattedResult";

function Tafsir() {
  const [ayah, setAyah] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(event) {
    event.preventDefault();

    if (!ayah.trim() || loading) {
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await getTafsir(ayah.trim());
      setResult(response.data);
    } catch (err) {
      setError(
        err.response?.data?.message ||
          "حصل خطأ أثناء جلب التفسير. حاول مرة أخرى."
      );
    } finally {
      setLoading(false);
    }
  }

  function handleInputKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSubmit(event);
    }
  }

  return (
    <main className="service-page">
      <div className="container service-page-content">
        <h1>التفسير</h1>
        <p className="page-description">
          ابحث عن الآية التي تريد معرفة تفسيرها.
        </p>

        <section className="tool-panel">
          <div className="tool-heading">
            <span className="service-icon">
              <i className="bi bi-book-half"></i>
            </span>
            <div>
              <h2>ابحث عن آية</h2>
              <p>أدخل الآية أو النص المتاح للبحث.</p>
            </div>
          </div>

          <form onSubmit={handleSubmit}>
            <label htmlFor="tafsir-ayah">الآية أو النص</label>

            <textarea
              id="tafsir-ayah"
              value={ayah}
              onChange={function (event) {
                setAyah(event.target.value);
              }}
              onKeyDown={handleInputKeyDown}
              placeholder="اكتب الآية هنا..."
              rows="5"
              required
            />

            <button
              className="primary-button"
              type="submit"
              disabled={loading}
            >
              {loading ? "جاري البحث..." : "البحث عن التفسير"}
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
            <section className="api-result" aria-live="polite">
              <h2>نتيجة التفسير</h2>
              <FormattedResult data={result} />
            </section>
          )}
        </section>
      </div>
    </main>
  );
}

export default Tafsir;
