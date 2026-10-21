
import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import api from "../services/api";
import { useAuth } from "../context/AuthContext";
import "./ForgotPassword.css";

function maskEmail(email) {
  if (!email || !email.includes("@")) return "";

  const [name, ...domainParts] = email.split("@");
  const domain = domainParts.join("@");

  if (name.length <= 2) {
    return `${name[0] || ""}${"*".repeat(
      Math.max(name.length - 1, 0)
    )}@${domain}`;
  }

  return `${name[0]}${"*".repeat(name.length - 2)}${
    name[name.length - 1]
  }@${domain}`;
}

function ForgotPassword() {
  const { user } = useAuth();
  const navigate = useNavigate();

  const loggedInEmail = user?.email || user?.user?.email || "";
  const isLoggedIn = Boolean(loggedInEmail);

  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    if (isLoggedIn) {
      setEmail(loggedInEmail.trim().toLowerCase());
    } else {
      setEmail("");
    }
  }, [isLoggedIn, loggedInEmail]);

  async function handleSubmit(event) {
    event.preventDefault();

    const emailToSend = isLoggedIn
      ? loggedInEmail.trim().toLowerCase()
      : email.trim().toLowerCase();

    if (!emailToSend) {
      setError("من فضلك اكتب البريد الإلكتروني المرتبط بحسابك.");
      return;
    }

    setMessage("");
    setError("");
    setLoading(true);

    try {
      const response = await api.post("/auth/forgotpassword", {
        email: emailToSend,
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

  const returnPath = isLoggedIn ? "/profile" : "/signin";
  const returnText = isLoggedIn
    ? "العودة للبروفايل"
    : "العودة لتسجيل الدخول";

  return (
    <main className="forgot-page" dir="rtl">
      <div className="forgot-glow forgot-glow-one"></div>
      <div className="forgot-glow forgot-glow-two"></div>

      <section className="forgot-card">
        <Link to={returnPath} className="forgot-back">
          <i className="bi bi-arrow-right"></i>
          <span>{returnText}</span>
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
          {isLoggedIn
            ? "هنرسل رابط استعادة كلمة المرور إلى البريد الإلكتروني المرتبط بحسابك."
            : "اكتب البريد الإلكتروني المرتبط بحسابك وهنساعدك تستعيد الوصول لحسابك."}
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
                type={isLoggedIn ? "text" : "email"}
                name="email"
                placeholder="example@gmail.com"
                value={
                  isLoggedIn
                    ? maskEmail(loggedInEmail)
                    : email
                }
                onChange={(event) => {
                  if (!isLoggedIn) {
                    setEmail(event.target.value);
                  }
                }}
                readOnly={isLoggedIn}
                aria-readonly={isLoggedIn}
                dir="ltr"
                autoComplete="email"
                required
                disabled={loading}
                className={isLoggedIn ? "forgot-email-readonly" : ""}
              />

              {isLoggedIn && (
                <i className="bi bi-lock-fill forgot-email-lock"></i>
              )}
            </div>

            {isLoggedIn && (
              <p className="forgot-email-note">
                البريد الإلكتروني مرتبط بحسابك ولا يمكن تغييره من هنا.
              </p>
            )}

            <button
              type="submit"
              className="forgot-submit"
              disabled={loading || (isLoggedIn && !loggedInEmail)}
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
            onClick={() => navigate(returnPath)}
          >
            {returnText}
            <i className="bi bi-arrow-left"></i>
          </button>
        )}

        <div className="forgot-help">
          <i className="bi bi-info-circle"></i>
          <span>
            تأكد من إمكانية الوصول إلى بريدك الإلكتروني، وراجع مجلد الرسائل غير
            المرغوب فيها (Spam) لو الرسالة موصلتش.
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
