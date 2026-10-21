
import { useState } from "react";
import { Link, Navigate, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import api from "../services/api";
import "./Profile.css";

function Profile() {
  const { user, setUser, logout } = useAuth();
  const navigate = useNavigate();

  // بيانات الملف الشخصي
  const [firstName, setFirstName] = useState(user?.firstName || "");
  const [lastName, setLastName] = useState(user?.lastName || "");
  const [email, setEmail] = useState(user?.email || "");

  // إظهار وإخفاء نموذج تعديل الملف الشخصي
  const [isEditingProfile, setIsEditingProfile] = useState(false);
  const [profileLoading, setProfileLoading] = useState(false);
  const [profileError, setProfileError] = useState("");
  const [profileSuccess, setProfileSuccess] = useState("");

  // تغيير كلمة المرور
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");

  const [showCurrentPassword, setShowCurrentPassword] = useState(false);
  const [showNewPassword, setShowNewPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  const [loading, setLoading] = useState(false);
  const [deactivating, setDeactivating] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);

  const [currentPasswordError, setCurrentPasswordError] = useState("");
  const [formError, setFormError] = useState("");
  const [successMessage, setSuccessMessage] = useState("");
  const [pageError, setPageError] = useState("");

  if (!user) {
    return <Navigate to="/signin" replace />;
  }

  const displayName =
    [user.firstName, user.lastName].filter(Boolean).join(" ") ||
    user.name ||
    user.email ||
    "المستخدم";

  // فتح نموذج تعديل الملف الشخصي
  function handleOpenProfileEdit() {
    // تعبئة الحقول ببيانات الحساب الحالية
    setFirstName(user.firstName || "");
    setLastName(user.lastName || "");
    setEmail(user.email || "");

    setProfileError("");
    setProfileSuccess("");
    setIsEditingProfile(true);
  }

  // إلغاء التعديل وإخفاء النموذج
  function handleCancelProfileEdit() {
    setFirstName(user.firstName || "");
    setLastName(user.lastName || "");
    setEmail(user.email || "");

    setProfileError("");
    setProfileSuccess("");
    setIsEditingProfile(false);
  }

  // تعديل بيانات الحساب وحفظها
  async function handleUpdateProfile(event) {
    event.preventDefault();

    setProfileError("");
    setProfileSuccess("");

    const cleanFirstName = firstName.trim();
    const cleanLastName = lastName.trim();
    const cleanEmail = email.trim().toLowerCase();

    if (!cleanFirstName || !cleanLastName || !cleanEmail) {
      setProfileError(
        "من فضلك املأ الاسم الأول واسم العائلة والبريد الإلكتروني."
      );
      return;
    }

    if (cleanFirstName.length > 50 || cleanLastName.length > 50) {
      setProfileError("يجب ألا يزيد كل جزء من الاسم عن 50 حرفًا.");
      return;
    }

    const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

    if (!emailPattern.test(cleanEmail)) {
      setProfileError("من فضلك أدخل بريدًا إلكترونيًا صحيحًا.");
      return;
    }

    const userId = user.id || user._id;

    if (!userId) {
      setProfileError(
        "تعذر تحديد حساب المستخدم. سجّل الدخول مرة أخرى."
      );
      return;
    }

    setProfileLoading(true);

    try {
      const response = await api.patch(`/users/${userId}`, {
        firstName: cleanFirstName,
        lastName: cleanLastName,
        email: cleanEmail,
      });

      const updatedUser = response.data?.data;

      if (!updatedUser) {
        throw new Error("لم يتم استلام بيانات الحساب بعد التحديث.");
      }

      // تحديث بيانات المستخدم في AuthContext
      setUser({
        ...user,
        ...updatedUser,
        id: updatedUser.id || updatedUser._id || userId,
        firstName: updatedUser.firstName ?? cleanFirstName,
        lastName: updatedUser.lastName ?? cleanLastName,
        email: String(updatedUser.email ?? cleanEmail)
          .trim()
          .toLowerCase(),
      });

      // تحديث الحقول بالقيم المحفوظة
      setFirstName(updatedUser.firstName ?? cleanFirstName);
      setLastName(updatedUser.lastName ?? cleanLastName);
      setEmail(
        String(updatedUser.email ?? cleanEmail).trim().toLowerCase()
      );

      setProfileSuccess("تم تحديث بيانات حسابك بنجاح.");

      // إخفاء نموذج التعديل بعد نجاح الحفظ
      setIsEditingProfile(false);
    } catch (error) {
      const responseMessage =
        error.response?.data?.message ||
        error.response?.data?.error ||
        "";

      const normalizedMessage = String(responseMessage).toLowerCase();

      if (
        normalizedMessage.includes("duplicate") ||
        normalizedMessage.includes("already exists") ||
        normalizedMessage.includes("email already") ||
        (normalizedMessage.includes("email") &&
          normalizedMessage.includes("exist"))
      ) {
        setProfileError("هذا البريد الإلكتروني مستخدم بالفعل.");
      } else {
        setProfileError(
          responseMessage ||
            (error.message === "Network Error"
              ? "تعذر الاتصال بالخادم. تأكد من تشغيل الباك إند."
              : error.message ||
                "تعذر تحديث بيانات الحساب. حاول مرة أخرى.")
        );
      }
    } finally {
      setProfileLoading(false);
    }
  }

  // تحديث حقول كلمة المرور
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

  function clearMessages() {
    setCurrentPasswordError("");
    setFormError("");
    setSuccessMessage("");
    setPageError("");
  }

  // تغيير كلمة المرور
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
        currentPassword,
        newPassword,
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

  // تسجيل الخروج
  async function handleLogout() {
    setPageError("");

    try {
      await logout();
      navigate("/signin", { replace: true });
    } catch {
      setPageError("تعذر تسجيل الخروج. حاول مرة أخرى.");
    }
  }

  // تعطيل الحساب
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

        {/* معلومات الحساب */}
        <section className="profile-card">
          <div className="profile-card-heading">
            <div className="profile-section-icon">
              <i className="bi bi-person-vcard" aria-hidden="true"></i>
            </div>

            <div>
              <h2>معلومات الحساب</h2>
              <p>عرض وإدارة بياناتك الشخصية</p>
            </div>
          </div>

          {/* البيانات الحالية */}
          <div className="profile-info">
            <div className="profile-info-item">
              <span className="profile-info-label">الاسم الحالي</span>
              <span className="profile-info-value">
                {displayName}
              </span>
            </div>

            <div className="profile-info-item">
              <span className="profile-info-label">
                البريد الإلكتروني
              </span>
              <span className="profile-info-value" dir="ltr">
                {user.email || "غير متوفر"}
              </span>
            </div>
          </div>

          {/* رسالة نجاح آخر تعديل */}
          {profileSuccess && (
            <div
              className="profile-alert profile-alert-success"
              role="status"
            >
              <i className="bi bi-check-circle" aria-hidden="true"></i>
              <span>{profileSuccess}</span>
            </div>
          )}

          {/* زر فتح نموذج التعديل */}
          {!isEditingProfile && (
            <button
              type="button"
              className="profile-primary-button profile-edit-toggle"
              onClick={handleOpenProfileEdit}
            >
              <i className="bi bi-pencil-square" aria-hidden="true"></i>
              تعديل الملف الشخصي
            </button>
          )}

          {/* نموذج التعديل يظهر عند الضغط على الزر فقط */}
          {isEditingProfile && (
            <form
              className="profile-edit-form"
              onSubmit={handleUpdateProfile}
              noValidate
            >
              <div className="profile-edit-fields">
                <div className="profile-form-group">
                  <label htmlFor="profileFirstName">الاسم الأول</label>
                  <input
                    id="profileFirstName"
                    type="text"
                    value={firstName}
                    onChange={(event) => {
                      setFirstName(event.target.value);
                      setProfileError("");
                      setProfileSuccess("");
                    }}
                    placeholder="أدخل الاسم الأول"
                    autoComplete="given-name"
                    maxLength={50}
                    disabled={profileLoading}
                    required
                  />
                </div>

                <div className="profile-form-group">
                  <label htmlFor="profileLastName">اسم العائلة</label>
                  <input
                    id="profileLastName"
                    type="text"
                    value={lastName}
                    onChange={(event) => {
                      setLastName(event.target.value);
                      setProfileError("");
                      setProfileSuccess("");
                    }}
                    placeholder="أدخل اسم العائلة"
                    autoComplete="family-name"
                    maxLength={50}
                    disabled={profileLoading}
                    required
                  />
                </div>

                <div className="profile-form-group profile-email-group">
                  <label htmlFor="profileEmail">
                    البريد الإلكتروني
                  </label>
                  <input
                    id="profileEmail"
                    type="email"
                    value={email}
                    onChange={(event) => {
                      setEmail(event.target.value);
                      setProfileError("");
                      setProfileSuccess("");
                    }}
                    placeholder="example@email.com"
                    autoComplete="email"
                    maxLength={254}
                    dir="ltr"
                    disabled={profileLoading}
                    required
                  />
                </div>
              </div>

              {profileError && (
                <div
                  className="profile-alert profile-alert-error"
                  role="alert"
                >
                  <i
                    className="bi bi-exclamation-circle"
                    aria-hidden="true"
                  ></i>
                  <span>{profileError}</span>
                </div>
              )}

              <div className="profile-edit-actions">
                <button
                  type="submit"
                  className="profile-primary-button"
                  disabled={profileLoading}
                >
                  {profileLoading ? (
                    <>
                      <span
                        className="profile-spinner"
                        aria-hidden="true"
                      ></span>
                      جارٍ حفظ البيانات...
                    </>
                  ) : (
                    <>
                      <i
                        className="bi bi-check2-circle"
                        aria-hidden="true"
                      ></i>
                      حفظ البيانات
                    </>
                  )}
                </button>

                <button
                  type="button"
                  className="profile-cancel-button"
                  onClick={handleCancelProfileEdit}
                  disabled={profileLoading}
                >
                  <i className="bi bi-x-circle" aria-hidden="true"></i>
                  إلغاء
                </button>
              </div>
            </form>
          )}
        </section>

        {/* تغيير كلمة المرور */}
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
                  <i
                    className="bi bi-shield-check"
                    aria-hidden="true"
                  ></i>
                  تغيير كلمة المرور
                </>
              )}
            </button>
          </form>
        </section>

        {/* إعدادات الحساب */}
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
                    {deactivating
                      ? "جارٍ التعطيل..."
                      : "نعم، تعطيل الحساب"}
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
