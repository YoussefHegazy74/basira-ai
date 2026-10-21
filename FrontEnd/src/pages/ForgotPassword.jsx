
import { useState } from "react";
import { Link } from "react-router-dom";
import api from "../services/api";
import "./ForgotPassword.css";

function ForgotPassword() {
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  async function handleSubmit(event) {
    event.preventDefault();

    setMessage("");
    setError("");
    setLoading(true);

    try {
      const response = await api.post("/auth/forgotpassword", {
        email: email.trim().toLowerCase(),
      });

      setMessage(
        response.data?.message ||
          "لو البريد الإلكتروني مسجل عندنا، هتوصلك تعليمات استعادة كلمة المرور."
      );
    } catch (err) {
      setError(
        err.response?.data?.message ||
          "حصلت مشكلة أثناء إرسال الطلب. حاول تاني بعد شوية."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="forgot-page" dir="rtl">
      <div className="forgot-glow forgot-glow-one"></div>
      <div className="forgot-glow forgot-glow-two"></div>

      <section className="forgot-card">
        <Link to="/signin" className="forgot-back">
          <i className="bi bi-arrow-right"></i>
          <span>العودة لتسجيل الدخول</span>
        </Link>

        <div className="forgot-brand">
          <div className="forgot-brand-icon">
            <i className="bi bi-eye-fill"></i>
          </div>
          <span>Basira AI</span>
        </div>

        <div className="forgot-main-icon">
          <i className="bi bi-shield-lock"></i>
        </div>

        <span className="forgot-eyebrow">استعادة الحساب</span>

        <h1>نسيت كلمة المرور؟</h1>

        <p className="forgot-description">
          ولا يهمك، ده بيحصل. اكتب البريد الإلكتروني المرتبط بحسابك
          وهنساعدك تستعيد الوصول لحسابك.
        </p>

        {error && (
          <div className="forgot-alert forgot-error" role="alert">
            <i className="bi bi-exclamation-circle-fill"></i>
            <span>{error}</span>
          </div>
        )}

        {message && (
          <div className="forgot-alert forgot-success" role="status">
            <i className="bi bi-check-circle-fill"></i>
            <span>{message}</span>
          </div>
        )}

        {!message && (
          <form className="forgot-form" onSubmit={handleSubmit}>
            <label htmlFor="forgot-email">البريد الإلكتروني</label>

            <div className="forgot-input-wrapper">
              <i className="bi bi-envelope"></i>

              <input
                id="forgot-email"
                type="email"
                name="email"
                placeholder="example@email.com"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                autoComplete="email"
                required
                disabled={loading}
              />
            </div>

            <button
              type="submit"
              className="forgot-submit"
              disabled={loading}
            >
              {loading ? (
                <>
                  <span className="forgot-spinner"></span>
                  جارٍ إرسال الطلب...
                </>
              ) : (
                <>
                  إرسال رابط الاستعادة
                  <i className="bi bi-arrow-left"></i>
                </>
              )}
            </button>
          </form>
        )}

        {message && (
          <button
            type="button"
            className="forgot-submit"
            onClick={() => {
              setMessage("");
              setEmail("");
            }}
          >
            إرسال طلب آخر
            <i className="bi bi-arrow-left"></i>
          </button>
        )}

        <div className="forgot-help">
          <i className="bi bi-info-circle"></i>
          <span>
            تأكد من كتابة البريد الإلكتروني الصحيح، وراجع مجلد الرسائل
            غير المرغوب فيها (Spam) لو الرسالة موصلتش.
          </span>
        </div>

        <div className="forgot-footer">
          <span>فاكر كلمة المرور؟</span>
          <Link to="/signin">سجّل دخولك</Link>
        </div>
      </section>

      <p className="forgot-copyright">
        © Basira AI — Your intelligent companion
      </p>
    </main>
  );
}

export default ForgotPassword;
