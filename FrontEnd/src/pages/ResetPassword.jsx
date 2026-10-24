import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import AuthShell from "../components/AuthShell";
import api from "../services/api";
import { arabicMessage } from "../utils/format";
import "../styles/auth.css";

function ResetPassword() {
  const { token } = useParams();
  const navigate = useNavigate();

  const [newPassword, setNewPassword] = useState("");
  const [newPasswordConfirm, setNewPasswordConfirm] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  useEffect(() => {
    if (!success) return undefined;
    const timer = window.setTimeout(() => navigate("/signin"), 2200);
    return () => window.clearTimeout(timer);
  }, [success, navigate]);

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");
    setSuccess("");

    if (newPassword.length < 8) {
      setError("اختر كلمة مرور من ثمانية أحرف على الأقل.");
      return;
    }

    if (newPassword !== newPasswordConfirm) {
      setError("كلمتا المرور غير متطابقتين.");
      return;
    }

    setLoading(true);

    try {
      const response = await api.patch(`/auth/resetpassword/${token}`, {
        newPassword,
        newPasswordConfirm,
      });

      setSuccess(
        arabicMessage(
          response.data?.message,
          "تم تغيير كلمة المرور. سننقلك إلى صفحة تسجيل الدخول.",
        ),
      );
    } catch (requestError) {
      setError(
        arabicMessage(
          requestError.response?.data?.message ||
            requestError.response?.data?.Message,
          "تعذّر تغيير كلمة المرور. ربما انتهت صلاحية الرابط، اطلب رابطًا جديدًا.",
        ),
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <AuthShell
      eyebrow="كلمة مرور جديدة"
      title="اختر كلمة مرور جديدة"
      description="استخدم كلمة مرور قوية لم يسبق استخدامها في حسابك."
      backTo="/signin"
      backLabel="العودة إلى تسجيل الدخول"
      asideTitle="خطوة بسيطة تفصلك عن حسابك"
      asideText="اختر كلمة مرور واضحة لك وصعبة على الآخرين، ولا تشارك رابط إعادة التعيين مع أي شخص."
      footer={
        <>
          تذكّرت كلمة المرور؟ <Link to="/signin">تسجيل الدخول</Link>
        </>
      }
    >
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

      <form className="auth-form" onSubmit={handleSubmit} noValidate>
        <div className="field">
          <label htmlFor="newPassword">كلمة المرور الجديدة</label>
          <div className="password-field">
            <input
              id="newPassword"
              type={showPassword ? "text" : "password"}
              autoComplete="new-password"
              placeholder="أدخل كلمة المرور الجديدة"
              value={newPassword}
              onChange={(event) => {
                setNewPassword(event.target.value);
                if (error) setError("");
              }}
              minLength={8}
              required
              disabled={loading || Boolean(success)}
            />
            <button
              type="button"
              className="password-toggle"
              onClick={() => setShowPassword((current) => !current)}
              aria-label={showPassword ? "إخفاء كلمة المرور" : "إظهار كلمة المرور"}
              aria-pressed={showPassword}
              disabled={loading || Boolean(success)}
            >
              <i
                className={showPassword ? "bi bi-eye-slash" : "bi bi-eye"}
                aria-hidden="true"
              ></i>
            </button>
          </div>
        </div>

        <div className="field">
          <label htmlFor="newPasswordConfirm">تأكيد كلمة المرور</label>
          <div className="password-field">
            <input
              id="newPasswordConfirm"
              type={showConfirm ? "text" : "password"}
              autoComplete="new-password"
              placeholder="أعد كتابة كلمة المرور"
              value={newPasswordConfirm}
              onChange={(event) => {
                setNewPasswordConfirm(event.target.value);
                if (error) setError("");
              }}
              minLength={8}
              required
              disabled={loading || Boolean(success)}
            />
            <button
              type="button"
              className="password-toggle"
              onClick={() => setShowConfirm((current) => !current)}
              aria-label={
                showConfirm ? "إخفاء تأكيد كلمة المرور" : "إظهار تأكيد كلمة المرور"
              }
              aria-pressed={showConfirm}
              disabled={loading || Boolean(success)}
            >
              <i
                className={showConfirm ? "bi bi-eye-slash" : "bi bi-eye"}
                aria-hidden="true"
              ></i>
            </button>
          </div>
        </div>

        <p className="field-hint">
          ثمانية أحرف على الأقل. يُفضّل أن تجمع بين الحروف والأرقام.
        </p>

        <button
          className="btn btn-primary"
          type="submit"
          disabled={loading || Boolean(success)}
          aria-busy={loading}
        >
          {loading ? (
            <>
              <span className="spinner" aria-hidden="true"></span>
              جارٍ تغيير كلمة المرور…
            </>
          ) : (
            "حفظ كلمة المرور"
          )}
        </button>
      </form>
    </AuthShell>
  );
}

export default ResetPassword;
