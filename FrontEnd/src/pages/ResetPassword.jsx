import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import api from "../services/api";
import "./ResetPassword.css";


function ResetPassword() {
  const { token } = useParams();
  const navigate = useNavigate();

  const [newPassword, setNewPassword] = useState("");
  const [newPasswordConfirm, setNewPasswordConfirm] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    setSuccess("");

    if (newPassword !== newPasswordConfirm) {
      setError("كلمتا المرور غير متطابقتين.");
      return;
    }

    setLoading(true);

    try {
      const response = await api.patch(
        `/auth/resetpassword/${token}`,
        {
          newPassword: newPassword,
          newPasswordConfirm: newPasswordConfirm,
        }
      );

      setSuccess(
        response.data.message || "تم تغيير كلمة المرور بنجاح."
      );

      setTimeout(() => {
        navigate("/signin");
      }, 2000);
    } catch (err) {
      setError(
        err.response?.data?.message ||
          err.response?.data?.Message ||
          "تعذر تغيير كلمة المرور. ربما انتهت صلاحية الرابط."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="reset-page">
      <section className="reset-card">
        <div className="reset-logo">B</div>

        <h1>إنشاء كلمة مرور جديدة</h1>

        <p className="reset-description">
          اختر كلمة مرور قوية لحماية حسابك في Basira AI.
        </p>

        {error && (
          <div className="reset-alert reset-error" role="alert">
            {error}
          </div>
        )}

        {success && (
          <div className="reset-alert reset-success" role="status">
            {success}
          </div>
        )}

        <form onSubmit={handleSubmit}>
          <div className="reset-field">
            <label htmlFor="newPassword">كلمة المرور الجديدة</label>
            <input
              id="newPassword"
              type="password"
              autoComplete="new-password"
              placeholder="أدخل كلمة المرور الجديدة"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              minLength={8}
              required
              disabled={loading || Boolean(success)}
            />
          </div>

          <div className="reset-field">
            <label htmlFor="newPasswordConfirm">
              تأكيد كلمة المرور
            </label>
            <input
              id="newPasswordConfirm"
              type="password"
              autoComplete="new-password"
              placeholder="أعد كتابة كلمة المرور"
              value={newPasswordConfirm}
              onChange={(e) => setNewPasswordConfirm(e.target.value)}
              minLength={8}
              required
              disabled={loading || Boolean(success)}
            />
          </div>

          <p className="reset-hint">
            يجب أن تحتوي كلمة المرور على 8 أحرف على الأقل،
            وحرف كبير وحرف صغير ورقم ورمز خاص.
          </p>

          <button
            className="reset-submit"
            type="submit"
            disabled={loading || Boolean(success)}
          >
            {loading ? "جارٍ تغيير كلمة المرور..." : "تغيير كلمة المرور"}
          </button>
        </form>

        <div className="reset-footer">
          <Link to="/signin">العودة إلى تسجيل الدخول</Link>
        </div>
      </section>
    </main>
  );
}

export default ResetPassword;