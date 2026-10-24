import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import AuthShell from "../components/AuthShell";
import api from "../services/api";
import { useAuth } from "../context/AuthContext";
import { arabicMessage } from "../utils/format";
import "../styles/auth.css";

function maskEmail(email) {
  if (!email || !email.includes("@")) return "";
  const [name, ...domainParts] = email.split("@");
  const domain = domainParts.join("@");

  if (name.length <= 2) {
    return `${name[0] || ""}${"*".repeat(Math.max(name.length - 1, 0))}@${domain}`;
  }

  return `${name[0]}${"*".repeat(name.length - 2)}${name[name.length - 1]}@${domain}`;
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
    setEmail(isLoggedIn ? loggedInEmail.trim().toLowerCase() : "");
  }, [isLoggedIn, loggedInEmail]);

  async function handleSubmit(event) {
    event.preventDefault();
    setMessage("");
    setError("");

    const emailToSend = isLoggedIn
      ? loggedInEmail.trim().toLowerCase()
      : email.trim().toLowerCase();

    if (!emailToSend) {
      setError("أدخل البريد الإلكتروني المرتبط بحسابك.");
      return;
    }

    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(emailToSend)) {
      setError("تحقق من صيغة البريد الإلكتروني ثم حاول مرة أخرى.");
      return;
    }

    setLoading(true);

    try {
      const response = await api.post("/auth/forgotpassword", {
        email: emailToSend,
      });

      setMessage(
        arabicMessage(
          response.data?.message,
          "إن كان البريد مسجلًا، فستصلك رسالة تحتوي على خطوات استعادة كلمة المرور.",
        ),
      );
    } catch (requestError) {
      setError(
        arabicMessage(
          requestError.response?.data?.message,
          "تعذّر إرسال الطلب الآن. حاول مرة أخرى بعد قليل.",
        ),
      );
    } finally {
      setLoading(false);
    }
  }

  const returnPath = isLoggedIn ? "/profile" : "/signin";
  const returnText = isLoggedIn
    ? "العودة إلى الملف الشخصي"
    : "العودة إلى تسجيل الدخول";

  return (
    <AuthShell
      eyebrow="استعادة الحساب"
      title="نسيت كلمة المرور؟"
      description={
        isLoggedIn
          ? "سنرسل رابط إعادة التعيين إلى البريد المرتبط بحسابك."
          : "أدخل بريدك الإلكتروني وسنساعدك على استعادة الوصول إلى حسابك."
      }
      backTo={returnPath}
      backLabel={returnText}
      footer={
        <>
          تذكّر كلمة المرور؟{" "}
          <Link to={returnPath}>
            {isLoggedIn ? "العودة إلى حسابك" : "تسجيل الدخول"}
          </Link>
        </>
      }
    >
      {error ? (
        <div className="alert alert-error" role="alert">
          <i className="bi bi-exclamation-circle" aria-hidden="true"></i>
          <span>{error}</span>
        </div>
      ) : null}

      {message ? (
        <>
          <div className="alert alert-success" role="status">
            <i className="bi bi-check-circle" aria-hidden="true"></i>
            <span>{message}</span>
          </div>

          <button
            type="button"
            className="btn btn-primary btn-block"
            onClick={() => navigate(returnPath)}
          >
            {returnText}
            <i className="bi bi-arrow-left" aria-hidden="true"></i>
          </button>
        </>
      ) : (
        <form className="auth-form" onSubmit={handleSubmit} noValidate>
          <div className="field">
            <label htmlFor="forgot-email">البريد الإلكتروني</label>
            <input
              id="forgot-email"
              type={isLoggedIn ? "text" : "email"}
              name="email"
              placeholder="أدخل بريدك الإلكتروني"
              value={isLoggedIn ? maskEmail(loggedInEmail) : email}
              onChange={(event) => {
                if (!isLoggedIn) setEmail(event.target.value);
                if (error) setError("");
              }}
              readOnly={isLoggedIn}
              aria-readonly={isLoggedIn}
              dir="ltr"
              autoComplete="email"
              disabled={loading}
              required
            />
            {isLoggedIn ? (
              <span className="field-hint">
                البريد مرتبط بحسابك، ولا يمكن تغييره من هنا.
              </span>
            ) : null}
          </div>

          <button
            className="btn btn-primary"
            type="submit"
            disabled={loading || (isLoggedIn && !loggedInEmail)}
            aria-busy={loading}
          >
            {loading ? (
              <>
                <span className="spinner" aria-hidden="true"></span>
                جارٍ إرسال الطلب…
              </>
            ) : (
              <>
                إرسال رابط الاستعادة
                <i className="bi bi-arrow-left" aria-hidden="true"></i>
              </>
            )}
          </button>
        </form>
      )}

      <div className="auth-note">
        <i className="bi bi-info-circle" aria-hidden="true"></i>
        <span>
          قد تصل الرسالة إلى مجلد الرسائل غير المرغوبة. تأكد أيضًا من كتابة
          البريد الصحيح.
        </span>
      </div>
    </AuthShell>
  );
}

export default ForgotPassword;
