
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import api from "../services/api";

function SignIn() {
  const navigate = useNavigate();
  const { setUser } = useAuth();

  const [form, setForm] = useState({
    email: "",
    password: "",
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function handleChange(event) {
    const { name, value } = event.target;
    setForm(function (previous) {
      return { ...previous, [name]: value };
    });
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");
    setLoading(true);

    try {
      const response = await api.post("/auth/login", {
        email: form.email.trim().toLowerCase(),
        password: form.password,
      });

      // The backend must return the authenticated user's safe public data.
      const loggedInUser = response.data?.user;

      if (!loggedInUser) {
        setError(
          "تم تسجيل الدخول من الخادم، لكن بيانات المستخدم غير موجودة في الاستجابة. يلزم تعديل استجابة الـ Backend."
        );
        return;
      }

      setUser(loggedInUser);
      navigate("/", { replace: true });
    } catch (requestError) {
      setError(
        requestError.response?.data?.message ||
          requestError.response?.data?.Message ||
          "تعذّر تسجيل الدخول. راجع البريد الإلكتروني وكلمة المرور وحاول مرة أخرى."
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
          <h1>أهلًا بعودتك</h1>
          <p>سجّل دخولك للمتابعة واستخدام خدمات بصيرة.</p>
        </div>

        {error && (
          <div className="auth-alert" role="alert">
            {error}
          </div>
        )}

        <form className="auth-form" onSubmit={handleSubmit}>
          <div className="auth-field">
            <label htmlFor="signin-email">البريد الإلكتروني</label>
            <input
              id="signin-email"
              type="email"
              name="email"
              autoComplete="email"
              placeholder="name@example.com"
              value={form.email}
              onChange={handleChange}
              required
            />
          </div>

          <div className="auth-field">
            <label htmlFor="signin-password">كلمة المرور</label>
            <input
              id="signin-password"
              type="password"
              name="password"
              autoComplete="current-password"
              placeholder="أدخل كلمة المرور"
              value={form.password}
              onChange={handleChange}
              required
            />
          </div>

          <div className="auth-form-extra">
            <Link to="/forgot-password">نسيت كلمة المرور؟</Link>
          </div>

          <button
            className="auth-submit"
            type="submit"
            disabled={loading}
          >
            {loading ? "جارٍ تسجيل الدخول..." : "تسجيل الدخول"}
          </button>
        </form>

        <p className="auth-switch">
          ليس لديك حساب؟ <Link to="/signup">أنشئ حسابًا جديدًا</Link>
        </p>
      </section>
    </main>
  );
}

export default SignIn;
