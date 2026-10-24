import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import AuthShell from "../components/AuthShell";
import { useAuth } from "../context/AuthContext";
import api from "../services/api";
import { arabicMessage } from "../utils/format";
import "../styles/auth.css";

function SignIn() {
  const navigate = useNavigate();
  const { setUser } = useAuth();

  const [form, setForm] = useState({ email: "", password: "" });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showPassword, setShowPassword] = useState(false);

  function handleChange(event) {
    const { name, value } = event.target;
    setForm((previous) => ({ ...previous, [name]: value }));
    if (error) setError("");
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");

    const email = form.email.trim().toLowerCase();
    const password = form.password;

    if (!email || !password) {
      setError("أدخل بريدك الإلكتروني وكلمة المرور للمتابعة.");
      return;
    }

    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      setError("تحقق من صيغة البريد الإلكتروني ثم حاول مرة أخرى.");
      return;
    }

    setLoading(true);

    try {
      const response = await api.post("/auth/login", { email, password });
      const loggedInUser = response.data?.user;

      if (!loggedInUser) {
        setError("تعذّر إكمال تسجيل الدخول. حاول مرة أخرى.");
        return;
      }

      setUser(loggedInUser);
      navigate("/", { replace: true });
    } catch (requestError) {
      setError(
        arabicMessage(
          requestError.response?.data?.message ||
            requestError.response?.data?.Message,
          "تعذّر تسجيل الدخول. راجع البريد الإلكتروني وكلمة المرور وحاول مرة أخرى.",
        ),
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <AuthShell
      eyebrow="تسجيل الدخول"
      title="أهلًا بعودتك"
      description="سجّل دخولك لمتابعة محادثاتك والوصول إلى أدواتك."
      footer={
        <>
          ليس لديك حساب؟ <Link to="/signup">أنشئ حسابًا جديدًا</Link>
        </>
      }
    >
      {error ? (
        <div className="alert alert-error" role="alert">
          <i className="bi bi-exclamation-circle" aria-hidden="true"></i>
          <span>{error}</span>
        </div>
      ) : null}

      <form className="auth-form" onSubmit={handleSubmit} noValidate>
        <div className="field">
          <label htmlFor="signin-email">البريد الإلكتروني</label>
          <input
            id="signin-email"
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
          <label htmlFor="signin-password">كلمة المرور</label>
          <div className="password-field">
            <input
              id="signin-password"
              type={showPassword ? "text" : "password"}
              name="password"
              autoComplete="current-password"
              placeholder="أدخل كلمة المرور"
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
        </div>

        <div className="auth-meta">
          <Link to="/forgot-password">نسيت كلمة المرور؟</Link>
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
              جارٍ تسجيل الدخول…
            </>
          ) : (
            "تسجيل الدخول"
          )}
        </button>
      </form>
    </AuthShell>
  );
}

export default SignIn;
