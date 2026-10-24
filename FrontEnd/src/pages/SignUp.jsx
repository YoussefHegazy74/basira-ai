import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import AuthShell from "../components/AuthShell";
import api from "../services/api";
import { arabicMessage, formatNumber } from "../utils/format";
import "../styles/auth.css";

const EMPTY_FORM = {
  firstName: "",
  lastName: "",
  email: "",
  password: "",
  passwordconfirm: "",
};

function SignUp() {
  const navigate = useNavigate();

  const [step, setStep] = useState("register");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [form, setForm] = useState(EMPTY_FORM);
  const [verificationToken, setVerificationToken] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  function handleChange(event) {
    const { name, value } = event.target;
    setForm((previous) => ({ ...previous, [name]: value }));
    if (error) setError("");
  }

  async function handleRegister(event) {
    event.preventDefault();
    setError("");
    setSuccess("");

    const firstName = form.firstName.trim();
    const lastName = form.lastName.trim();
    const email = form.email.trim().toLowerCase();

    if (!firstName || !lastName || !email || !form.password) {
      setError("أكمل بيانات الحساب المطلوبة للمتابعة.");
      return;
    }

    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      setError("تحقق من صيغة البريد الإلكتروني ثم حاول مرة أخرى.");
      return;
    }

    if (form.password.length < 8) {
      setError("اختر كلمة مرور من ثمانية أحرف على الأقل.");
      return;
    }

    if (form.password !== form.passwordconfirm) {
      setError("كلمتا المرور غير متطابقتين.");
      return;
    }

    setLoading(true);

    try {
      await api.post("/auth/register", {
        firstName,
        lastName,
        email,
        password: form.password,
        passwordconfirm: form.passwordconfirm,
      });

      setStep("verify");
      setSuccess("أرسلنا رمز التحقق إلى بريدك الإلكتروني. أدخله لإكمال إنشاء الحساب.");
    } catch (requestError) {
      setError(
        arabicMessage(
          requestError.response?.data?.message ||
            requestError.response?.data?.Message,
          "تعذّر إنشاء الحساب. راجع البيانات وحاول مرة أخرى.",
        ),
      );
    } finally {
      setLoading(false);
    }
  }

  async function handleVerify(event) {
    event.preventDefault();
    setError("");
    setSuccess("");

    const code = verificationToken.trim();

    if (!/^\d{8}$/.test(code)) {
      setError("أدخل رمز التحقق المكوّن من ثمانية أرقام.");
      return;
    }

    setLoading(true);

    try {
      await api.post("/auth/verifyemail", {
        email: form.email.trim().toLowerCase(),
        verificationToken: code,
      });

      navigate("/signin", {
        replace: true,
        state: { email: form.email.trim().toLowerCase() },
      });
    } catch (requestError) {
      setError(
        arabicMessage(
          requestError.response?.data?.message ||
            requestError.response?.data?.Message,
          "رمز التحقق غير صحيح أو انتهت صلاحيته. حاول مرة أخرى.",
        ),
      );
    } finally {
      setLoading(false);
    }
  }

  function backToRegister() {
    setStep("register");
    setVerificationToken("");
    setError("");
    setSuccess("");
  }

  const isRegister = step === "register";

  return (
    <AuthShell
      eyebrow={isRegister ? "حساب جديد" : "الخطوة الثانية"}
      title={isRegister ? "أنشئ حسابك" : "تحقق من بريدك"}
      description={
        isRegister
          ? "احتفظ بمحادثاتك، وتابع أدواتك من أي جهاز."
          : "أدخل الرمز المرسل إلى بريدك لإكمال إنشاء الحساب."
      }
      footer={
        isRegister ? (
          <>
            لديك حساب بالفعل؟ <Link to="/signin">تسجيل الدخول</Link>
          </>
        ) : (
          <button type="button" className="auth-help-link" onClick={backToRegister}>
            العودة إلى بيانات الحساب
          </button>
        )
      }
    >
      <div className="auth-step">
        <span aria-hidden="true">{formatNumber(isRegister ? 1 : 2)}</span>
        الخطوة {formatNumber(isRegister ? 1 : 2)} من {formatNumber(2)}
      </div>

      {error ? (
        <div className="alert alert-error" role="alert">
          <i className="bi bi-exclamation-circle" aria-hidden="true"></i>
          <span>{error}</span>
        </div>
      ) : null}

      {success ? (
        <div className="alert alert-success" role="status">
          <i className="bi bi-check-circle" aria-hidden="true"></i>
          <span>{success}</span>
        </div>
      ) : null}

      {isRegister ? (
        <form className="auth-form" onSubmit={handleRegister} noValidate>
          <div className="field-grid">
            <div className="field">
              <label htmlFor="signup-first-name">الاسم الأول</label>
              <input
                id="signup-first-name"
                name="firstName"
                autoComplete="given-name"
                placeholder="اسمك الأول"
                value={form.firstName}
                onChange={handleChange}
                disabled={loading}
                required
              />
            </div>

            <div className="field">
              <label htmlFor="signup-last-name">اسم العائلة</label>
              <input
                id="signup-last-name"
                name="lastName"
                autoComplete="family-name"
                placeholder="اسم العائلة"
                value={form.lastName}
                onChange={handleChange}
                disabled={loading}
                required
              />
            </div>
          </div>

          <div className="field">
            <label htmlFor="signup-email">البريد الإلكتروني</label>
            <input
              id="signup-email"
              type="email"
              name="email"
              autoComplete="email"
              inputMode="email"
              dir="ltr"
              placeholder="أدخل بريدك الإلكتروني"
              value={form.email}
              onChange={handleChange}
              disabled={loading}
              required
            />
          </div>

          <div className="field">
            <label htmlFor="signup-password">كلمة المرور</label>
            <div className="password-field">
              <input
                id="signup-password"
                type={showPassword ? "text" : "password"}
                name="password"
                autoComplete="new-password"
                placeholder="اختر كلمة مرور"
                value={form.password}
                onChange={handleChange}
                disabled={loading}
                required
              />
              <button
                type="button"
                className="password-toggle"
                onClick={() => setShowPassword((current) => !current)}
                aria-label={showPassword ? "إخفاء كلمة المرور" : "إظهار كلمة المرور"}
                aria-pressed={showPassword}
                disabled={loading}
              >
                <i
                  className={showPassword ? "bi bi-eye-slash" : "bi bi-eye"}
                  aria-hidden="true"
                ></i>
              </button>
            </div>
            <span className="field-hint">ثمانية أحرف على الأقل.</span>
          </div>

          <div className="field">
            <label htmlFor="signup-confirm-password">تأكيد كلمة المرور</label>
            <div className="password-field">
              <input
                id="signup-confirm-password"
                type={showConfirmPassword ? "text" : "password"}
                name="passwordconfirm"
                autoComplete="new-password"
                placeholder="أعد كتابة كلمة المرور"
                value={form.passwordconfirm}
                onChange={handleChange}
                disabled={loading}
                required
              />
              <button
                type="button"
                className="password-toggle"
                onClick={() => setShowConfirmPassword((current) => !current)}
                aria-label={
                  showConfirmPassword
                    ? "إخفاء تأكيد كلمة المرور"
                    : "إظهار تأكيد كلمة المرور"
                }
                aria-pressed={showConfirmPassword}
                disabled={loading}
              >
                <i
                  className={showConfirmPassword ? "bi bi-eye-slash" : "bi bi-eye"}
                  aria-hidden="true"
                ></i>
              </button>
            </div>
          </div>

          <button
            className="btn btn-primary"
            type="submit"
            disabled={loading}
            aria-busy={loading}
          >
            {loading ? (
              <>
                <span className="spinner" aria-hidden="true"></span>
                جارٍ إنشاء الحساب…
              </>
            ) : (
              "إنشاء الحساب"
            )}
          </button>
        </form>
      ) : (
        <form className="auth-form" onSubmit={handleVerify} noValidate>
          <div className="field auth-code-input">
            <label htmlFor="verification-code">رمز التحقق</label>
            <input
              id="verification-code"
              type="text"
              inputMode="numeric"
              autoComplete="one-time-code"
              maxLength={8}
              dir="ltr"
              value={verificationToken}
              onChange={(event) =>
                setVerificationToken(
                  event.target.value.replace(/\D/g, "").slice(0, 8),
                )
              }
              placeholder="أدخل الرمز المكوّن من ثمانية أرقام"
              disabled={loading}
              required
            />
            <span className="field-hint">
              أرسلنا الرمز إلى {form.email}. راجع صندوق الوارد والرسائل غير
              المرغوبة.
            </span>
          </div>

          <button
            className="btn btn-primary"
            type="submit"
            disabled={loading}
            aria-busy={loading}
          >
            {loading ? (
              <>
                <span className="spinner" aria-hidden="true"></span>
                جارٍ التحقق…
              </>
            ) : (
              "تأكيد البريد الإلكتروني"
            )}
          </button>

          <button
            className="btn btn-secondary auth-alt-action"
            type="button"
            onClick={backToRegister}
            disabled={loading}
          >
            العودة إلى بيانات الحساب
          </button>
        </form>
      )}
    </AuthShell>
  );
}

export default SignUp;
