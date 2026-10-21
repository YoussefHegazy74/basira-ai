
// src/pages/Profile.jsx

import { useState } from "react";
import { Link, Navigate, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import api from "../services/api";
import "./Profile.css";

function Profile() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");

  const [showCurrentPassword, setShowCurrentPassword] = useState(false);
  const [showNewPassword, setShowNewPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  const [loading, setLoading] = useState(false);
  const [deactivating, setDeactivating] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);

  // أخطاء تغيير كلمة المرور
  const [currentPasswordError, setCurrentPasswordError] = useState("");
  const [formError, setFormError] = useState("");
  const [successMessage, setSuccessMessage] = useState("");

  // رسائل العمليات الأخرى
  const [pageError, setPageError] = useState("");

  if (!user) {
    return <Navigate to="/signin" replace />;
  }

  const displayName =
    [user.firstName, user.lastName].filter(Boolean).join(" ") ||
    user.name ||
    user.email ||
    "المستخدم";

  function clearMessages() {
    setCurrentPasswordError("");
    setFormError("");
    setSuccessMessage("");
    setPageError("");
  }

  function handlePasswordChange(event) {
    const { name, value } = event.target;

    if (name === "currentPassword") {
      setCurrentPassword(value);
      setCurrentPasswordError("");
    }

    if (name === "newPassword") {
      setNewPassword(value);
      setFormError("");
    }

    if (name === "confirmPassword") {
      setConfirmPassword(value);
      setFormError("");
    }

    setSuccessMessage("");
    setPageError("");
  }

  async function handleChangePassword(event) {
    event.preventDefault();
    clearMessages();

    if (!currentPassword.trim()) {
      setCurrentPasswordError("من فضلك أدخل كلمة المرور الحالية.");
      return;
    }

    if (!newPassword) {
      setFormError("من فضلك أدخل كلمة المرور الجديدة.");
      return;
    }

    if (newPassword.length < 8) {
      setFormError("كلمة المرور الجديدة يجب أن تكون 8 أحرف على الأقل.");
      return;
    }

    if (newPassword !== confirmPassword) {
      setFormError("كلمتا المرور الجديدتان غير متطابقتين.");
      return;
    }

    if (currentPassword === newPassword) {
      setFormError("كلمة المرور الجديدة يجب أن تختلف عن الحالية.");
      return;
    }

    setLoading(true);

    try {
      await api.patch("/auth/changepassword", {
        currentPassword: currentPassword,
        newPassword: newPassword,
      });

      setCurrentPassword("");
      setNewPassword("");
      setConfirmPassword("");

      setShowCurrentPassword(false);
      setShowNewPassword(false);
      setShowConfirmPassword(false);

      setSuccessMessage("تم تغيير كلمة المرور بنجاح.");
    } catch (error) {
      const responseMessage =
        error.response?.data?.message ||
        error.response?.data?.error ||
        "";

      const normalizedMessage = String(responseMessage).toLowerCase();

      const isCurrentPasswordError =
        normalizedMessage.includes("current password") ||
        normalizedMessage.includes("incorrect password") ||
        normalizedMessage.includes("wrong password") ||
        normalizedMessage.includes("password is incorrect") ||
        normalizedMessage.includes("invalid current password") ||
        normalizedMessage.includes("old password") ||
        responseMessage.includes("كلمة المرور الحالية") ||
        responseMessage.includes("كلمة المرور غير صحيحة") ||
        responseMessage.includes("كلمه المرور الحاليه");

      if (isCurrentPasswordError) {
        // رسالة خطأ الباسورد القديم تظهر أسفل حقله فقط
        setCurrentPasswordError(
          "كلمة المرور الحالية غير صحيحة، من فضلك حاول مرة أخرى."
        );
      } else {
        setFormError(
          responseMessage ||
            "تعذر تغيير كلمة المرور. من فضلك حاول مرة أخرى."
        );
      }
    } finally {
      setLoading(false);
    }
  }

  async function handleLogout() {
    setPageError("");

    try {
      await logout();
      navigate("/signin", { replace: true });
    } catch (error) {
      setPageError("تعذر تسجيل الخروج. حاول مرة أخرى.");
    }
  }

  async function handleDeactivateAccount() {
    setPageError("");
    setDeactivating(true);

    try {
      await api.patch("/auth/deactivate-account");

      await logout();
      navigate("/signin", { replace: true });
    } catch (error) {
      setPageError(
        error.response?.data?.message ||
          "تعذر تعطيل الحساب. من فضلك حاول مرة أخرى."
      );
      setShowDeleteConfirm(false);
    } finally {
      setDeactivating(false);
    }
  }

  return (
    <main className="profile-page" dir="rtl">
      <div className="profile-container">
        <header className="profile-header">
          <div className="profile-header-icon">
            <i className="bi bi-person-circle" aria-hidden="true"></i>
          </div>

          <h1>الملف الشخصي</h1>

          <p>إدارة حسابك وإعدادات الأمان في بصيرة AI</p>
        </header>

        {pageError && (
          <div className="profile-alert profile-alert-error" role="alert">
            <i className="bi bi-exclamation-circle" aria-hidden="true"></i>
            <span>{pageError}</span>
          </div>
        )}

        <section className="profile-card">
          <div className="profile-card-heading">
            <div className="profile-section-icon">
              <i className="bi bi-person-vcard" aria-hidden="true"></i>
            </div>

            <div>
              <h2>معلومات الحساب</h2>
              <p>معلومات حسابك الحالية</p>
            </div>
          </div>

          <div className="profile-info">
            <div className="profile-info-item">
              <span className="profile-info-label">الاسم</span>
              <span className="profile-info-value">{displayName}</span>
            </div>

            <div className="profile-info-item">
              <span className="profile-info-label">البريد الإلكتروني</span>
              <span className="profile-info-value">
                {user.email || "غير متوفر"}
              </span>
            </div>
          </div>
        </section>

        <section className="profile-card">
          <div className="profile-card-heading">
            <div className="profile-section-icon">
              <i className="bi bi-shield-lock" aria-hidden="true"></i>
            </div>

            <div>
              <h2>تغيير كلمة المرور</h2>
              <p>استخدم كلمة مرور قوية للحفاظ على أمان حسابك</p>
            </div>
          </div>

          <form
            className="profile-password-form"
            onSubmit={handleChangePassword}
            noValidate
          >
            <div className="profile-form-group">
              <label htmlFor="currentPassword">
                كلمة المرور الحالية
              </label>

              <div
                className={`profile-password-wrapper ${
                  currentPasswordError ? "has-error" : ""
                }`}
              >
                <input
                  id="currentPassword"
                  name="currentPassword"
                  type={showCurrentPassword ? "text" : "password"}
                  value={currentPassword}
                  onChange={handlePasswordChange}
                  placeholder="أدخل كلمة المرور الحالية"
                  autoComplete="current-password"
                  aria-invalid={Boolean(currentPasswordError)}
                  aria-describedby={
                    currentPasswordError
                      ? "currentPasswordError"
                      : undefined
                  }
                  disabled={loading}
                />

                <button
                  type="button"
                  className="profile-password-toggle"
                  onClick={() =>
                    setShowCurrentPassword(!showCurrentPassword)
                  }
                  aria-label={
                    showCurrentPassword
                      ? "إخفاء كلمة المرور الحالية"
                      : "إظهار كلمة المرور الحالية"
                  }
                >
                  <i
                    className={`bi ${
                      showCurrentPassword ? "bi-eye-slash" : "bi-eye"
                    }`}
                    aria-hidden="true"
                  ></i>
                </button>
              </div>

              {/* رسالة الخطأ هنا تحت حقل الباسورد القديم مباشرة */}
              {currentPasswordError && (
                <div
                  id="currentPasswordError"
                  className="profile-field-error"
                  role="alert"
                >
                  <i
                    className="bi bi-exclamation-circle"
                    aria-hidden="true"
                  ></i>
                  <span>{currentPasswordError}</span>
                </div>
              )}

              <div className="profile-forgot-password">
                <Link to="/forgot-password">نسيت كلمة المرور؟</Link>
              </div>
            </div>

            <div className="profile-form-group">
              <label htmlFor="newPassword">كلمة المرور الجديدة</label>

              <div className="profile-password-wrapper">
                <input
                  id="newPassword"
                  name="newPassword"
                  type={showNewPassword ? "text" : "password"}
                  value={newPassword}
                  onChange={handlePasswordChange}
                  placeholder="أدخل كلمة المرور الجديدة"
                  autoComplete="new-password"
                  disabled={loading}
                />

                <button
                  type="button"
                  className="profile-password-toggle"
                  onClick={() => setShowNewPassword(!showNewPassword)}
                  aria-label={
                    showNewPassword
                      ? "إخفاء كلمة المرور الجديدة"
                      : "إظهار كلمة المرور الجديدة"
                  }
                >
                  <i
                    className={`bi ${
                      showNewPassword ? "bi-eye-slash" : "bi-eye"
                    }`}
                    aria-hidden="true"
                  ></i>
                </button>
              </div>

              <small className="profile-input-hint">
                استخدم 8 أحرف على الأقل.
              </small>
            </div>

            <div className="profile-form-group">
              <label htmlFor="confirmPassword">
                تأكيد كلمة المرور الجديدة
              </label>

              <div className="profile-password-wrapper">
                <input
                  id="confirmPassword"
                  name="confirmPassword"
                  type={showConfirmPassword ? "text" : "password"}
                  value={confirmPassword}
                  onChange={handlePasswordChange}
                  placeholder="أعد كتابة كلمة المرور الجديدة"
                  autoComplete="new-password"
                  disabled={loading}
                />

                <button
                  type="button"
                  className="profile-password-toggle"
                  onClick={() =>
                    setShowConfirmPassword(!showConfirmPassword)
                  }
                  aria-label={
                    showConfirmPassword
                      ? "إخفاء تأكيد كلمة المرور"
                      : "إظهار تأكيد كلمة المرور"
                  }
                >
                  <i
                    className={`bi ${
                      showConfirmPassword ? "bi-eye-slash" : "bi-eye"
                    }`}
                    aria-hidden="true"
                  ></i>
                </button>
              </div>
            </div>

            {formError && (
              <div className="profile-alert profile-alert-error" role="alert">
                <i
                  className="bi bi-exclamation-circle"
                  aria-hidden="true"
                ></i>
                <span>{formError}</span>
              </div>
            )}

            {successMessage && (
              <div
                className="profile-alert profile-alert-success"
                role="status"
              >
                <i className="bi bi-check-circle" aria-hidden="true"></i>
                <span>{successMessage}</span>
              </div>
            )}

            <button
              type="submit"
              className="profile-primary-button"
              disabled={loading}
            >
              {loading ? (
                <>
                  <span
                    className="profile-spinner"
                    aria-hidden="true"
                  ></span>
                  جارٍ تغيير كلمة المرور...
                </>
              ) : (
                <>
                  <i className="bi bi-shield-check" aria-hidden="true"></i>
                  تغيير كلمة المرور
                </>
              )}
            </button>
          </form>
        </section>

        <section className="profile-card profile-account-actions">
          <div className="profile-card-heading">
            <div className="profile-section-icon">
              <i className="bi bi-gear" aria-hidden="true"></i>
            </div>

            <div>
              <h2>إعدادات الحساب</h2>
              <p>إدارة الجلسة والحساب</p>
            </div>
          </div>

          <button
            type="button"
            className="profile-secondary-button"
            onClick={handleLogout}
          >
            <i className="bi bi-box-arrow-right" aria-hidden="true"></i>
            تسجيل الخروج
          </button>

          <div className="profile-danger-zone">
            <h3>تعطيل الحساب</h3>
            <p>
              استخدم هذا الخيار لتعطيل حسابك. قد لا تتمكن من استخدام
              الحساب بعد تعطيله.
            </p>

            {!showDeleteConfirm ? (
              <button
                type="button"
                className="profile-danger-button"
                onClick={() => setShowDeleteConfirm(true)}
              >
                <i className="bi bi-person-x" aria-hidden="true"></i>
                تعطيل الحساب
              </button>
            ) : (
              <div className="profile-confirm-box">
                <p>هل أنت متأكد أنك تريد تعطيل حسابك؟</p>

                <div className="profile-confirm-actions">
                  <button
                    type="button"
                    className="profile-danger-button"
                    onClick={handleDeactivateAccount}
                    disabled={deactivating}
                  >
                    {deactivating ? "جارٍ التعطيل..." : "نعم، تعطيل الحساب"}
                  </button>

                  <button
                    type="button"
                    className="profile-cancel-button"
                    onClick={() => setShowDeleteConfirm(false)}
                    disabled={deactivating}
                  >
                    إلغاء
                  </button>
                </div>
              </div>
            )}
          </div>
        </section>
      </div>
    </main>
  );
}

export default Profile;
