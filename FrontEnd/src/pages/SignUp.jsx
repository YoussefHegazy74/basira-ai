
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import api from "../services/api";

function SignUp() {
  const navigate = useNavigate();

  const [step, setStep] = useState("register");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const [form, setForm] = useState({
    firstName: "",
    lastName: "",
    email: "",
    password: "",
    passwordconfirm: "",
  });

  const [verificationToken, setVerificationToken] = useState("");

  function handleChange(event) {
    const { name, value } = event.target;
    setForm(function (previous) {
      return { ...previous, [name]: value };
    });
  }

  async function handleRegister(event) {
    event.preventDefault();
    setError("");
    setSuccess("");

    if (form.password !== form.passwordconfirm) {
      setError("كلمتا المرور غير متطابقتين.");
      return;
    }

    setLoading(true);

    try {
      await api.post("/auth/register", {
        firstName: form.firstName.trim(),
        lastName: form.lastName.trim(),
        email: form.email.trim().toLowerCase(),
        password: form.password,
        passwordconfirm: form.passwordconfirm,
      });

      setStep("verify");
      setSuccess("تم إرسال كود التحقق إلى بريدك الإلكتروني.");
    } catch (requestError) {
      setError(
        requestError.response?.data?.message ||
          requestError.response?.data?.Message ||
          "تعذّر إنشاء الحساب. تأكد من البيانات وحاول مرة أخرى."
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
      setError("أدخل كود التحقق المكوّن من 8 أرقام.");
      return;
    }

    setLoading(true);

    try {
      await api.post("/auth/verifyemail", {
        email: form.email.trim().toLowerCase(),
        verificationToken: code,
      });

      setSuccess("تم تفعيل حسابك بنجاح. يمكنك تسجيل الدخول الآن.");

      navigate("/signin", {
        replace: true,
        state: { email: form.email.trim().toLowerCase() },
      });
    } catch (requestError) {
      setError(
        requestError.response?.data?.message ||
          requestError.response?.data?.Message ||
          "كود التحقق غير صحيح أو انتهت صلاحيته. حاول مرة أخرى."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="auth-page" dir="rtl">
      <section className="auth-card">
        <Link to="/" className="auth-logo">
          <i className="bi bi-eye-fill"></i>
          <span>بصيرة <small>Basira AI</small></span>
        </Link>

        <div className="auth-heading">
          <h1>{step === "register" ? "أنشئ حسابك" : "تحقق من بريدك"}</h1>
          <p>
            {step === "register"
              ? "انضم إلى بصيرة وابدأ باستخدام خدمات الذكاء الاصطناعي."
              : `أدخل كود التحقق المرسل إلى ${form.email}.`}
          </p>
        </div>

        {error && (
          <div className="auth-alert" role="alert">
            {error}
          </div>
        )}

        {success && (
          <div className="auth-alert" role="status">
            {success}
          </div>
        )}

        {step === "register" ? (
          <form className="auth-form" onSubmit={handleRegister}>
            <div className="auth-name-fields">
              <div className="auth-field">
                <label htmlFor="signup-first-name">الاسم الأول</label>
                <input
                  id="signup-first-name"
                  name="firstName"
                  autoComplete="given-name"
                  value={form.firstName}
                  onChange={handleChange}
                  placeholder="الاسم الأول"
                  required
                />
              </div>

              <div className="auth-field">
                <label htmlFor="signup-last-name">اسم العائلة</label>
                <input
                  id="signup-last-name"
                  name="lastName"
                  autoComplete="family-name"
                  value={form.lastName}
                  onChange={handleChange}
                  placeholder="اسم العائلة"
                  required
                />
              </div>
            </div>

            <div className="auth-field">
              <label htmlFor="signup-email">البريد الإلكتروني</label>
              <input
                id="signup-email"
                type="email"
                name="email"
                autoComplete="email"
                value={form.email}
                onChange={handleChange}
                placeholder="name@example.com"
                required
              />
            </div>

            <div className="auth-field">
              <label htmlFor="signup-password">كلمة المرور</label>
              <input
                id="signup-password"
                type="password"
                name="password"
                autoComplete="new-password"
                value={form.password}
                onChange={handleChange}
                placeholder="أنشئ كلمة مرور"
                required
              />
            </div>

            <div className="auth-field">
              <label htmlFor="signup-confirm-password">
                تأكيد كلمة المرور
              </label>
              <input
                id="signup-confirm-password"
                type="password"
                name="passwordconfirm"
                autoComplete="new-password"
                value={form.passwordconfirm}
                onChange={handleChange}
                placeholder="أعد كتابة كلمة المرور"
                required
              />
            </div>

            <button className="auth-submit" type="submit" disabled={loading}>
              {loading ? "جارٍ إنشاء الحساب..." : "إنشاء الحساب"}
            </button>
          </form>
        ) : (
          <form className="auth-form" onSubmit={handleVerify}>
            <div className="auth-field">
              <label htmlFor="verification-code">كود التحقق</label>
              <input
                id="verification-code"
                type="text"
                inputMode="numeric"
                autoComplete="one-time-code"
                maxLength={8}
                value={verificationToken}
                onChange={(event) =>
                  setVerificationToken(
                    event.target.value.replace(/\D/g, "").slice(0, 8)
                  )
                }
                placeholder="أدخل الكود المكوّن من 8 أرقام"
                required
              />
            </div>

            <button className="auth-submit" type="submit" disabled={loading}>
              {loading ? "جارٍ التحقق..." : "تأكيد البريد الإلكتروني"}
            </button>

            <button
              className="auth-secondary"
              type="button"
              onClick={() => {
                setStep("register");
                setError("");
                setSuccess("");
              }}
            >
              العودة إلى بيانات الحساب
            </button>
          </form>
        )}

        <p className="auth-switch">
          لديك حساب بالفعل؟ <Link to="/signin">تسجيل الدخول</Link>
        </p>
      </section>
    </main>
  );
}

export default SignUp;
