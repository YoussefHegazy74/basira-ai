import { useState } from "react";
import { Link, Navigate, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import api from "../services/api";
import { arabicMessage, initials } from "../utils/format";
import { ConfirmDialog } from "../components/Modal";
import "../styles/profile.css";

function Profile() {
  const { user, setUser, logout, authLoading } = useAuth();
  const navigate = useNavigate();

  const [firstName, setFirstName] = useState(user?.firstName || "");
  const [lastName, setLastName] = useState(user?.lastName || "");
  const [email, setEmail] = useState(user?.email || "");

  const [isEditingProfile, setIsEditingProfile] = useState(false);
  const [profileLoading, setProfileLoading] = useState(false);
  const [profileError, setProfileError] = useState("");
  const [profileSuccess, setProfileSuccess] = useState("");

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

  if (authLoading) {
    return (
      <div className="route-state" aria-live="polite">
        <span className="spinner" aria-hidden="true"></span>
        <p>جارٍ تحميل حسابك…</p>
      </div>
    );
  }

  if (!user) return <Navigate to="/signin" replace />;

  const displayName =
    [user.firstName, user.lastName].filter(Boolean).join(" ") ||
    user.name ||
    user.email ||
    "المستخدم";

  const roleLabel = user.role === "admin" ? "مدير النظام" : "مستخدم";

  function handleOpenProfileEdit() {
    setFirstName(user.firstName || "");
    setLastName(user.lastName || "");
    setEmail(user.email || "");
    setProfileError("");
    setProfileSuccess("");
    setIsEditingProfile(true);
  }

  function handleCancelProfileEdit() {
    setFirstName(user.firstName || "");
    setLastName(user.lastName || "");
    setEmail(user.email || "");
    setProfileError("");
    setProfileSuccess("");
    setIsEditingProfile(false);
  }

  async function handleUpdateProfile(event) {
    event.preventDefault();
    setProfileError("");
    setProfileSuccess("");

    const cleanFirstName = firstName.trim();
    const cleanLastName = lastName.trim();
    const cleanEmail = email.trim().toLowerCase();

    if (!cleanFirstName || !cleanLastName || !cleanEmail) {
      setProfileError("أكمل الاسم الأول واسم العائلة والبريد الإلكتروني.");
      return;
    }

    if (cleanFirstName.length > 50 || cleanLastName.length > 50) {
      setProfileError("يجب ألا يزيد كل جزء من الاسم عن خمسين حرفًا.");
      return;
    }

    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(cleanEmail)) {
      setProfileError("تحقق من صيغة البريد الإلكتروني ثم حاول مرة أخرى.");
      return;
    }

    const userId = user.id || user._id;
    if (!userId) {
      setProfileError("تعذّر تحديد حسابك. سجّل الدخول مرة أخرى.");
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
        throw new Error("لم تصل بيانات الحساب بعد التحديث.");
      }

      setUser({
        ...user,
        ...updatedUser,
        id: updatedUser.id || updatedUser._id || userId,
        firstName: updatedUser.firstName ?? cleanFirstName,
        lastName: updatedUser.lastName ?? cleanLastName,
        email: String(updatedUser.email ?? cleanEmail).trim().toLowerCase(),
      });

      setFirstName(updatedUser.firstName ?? cleanFirstName);
      setLastName(updatedUser.lastName ?? cleanLastName);
      setEmail(String(updatedUser.email ?? cleanEmail).trim().toLowerCase());
      setProfileSuccess("تم تحديث بيانات حسابك.");
      setIsEditingProfile(false);
    } catch (requestError) {
      const responseMessage =
        requestError.response?.data?.message ||
        requestError.response?.data?.error ||
        "";
      const normalized = String(responseMessage).toLowerCase();
      const isDuplicate =
        normalized.includes("duplicate") ||
        normalized.includes("already exists") ||
        normalized.includes("email already") ||
        (normalized.includes("email") && normalized.includes("exist"));

      setProfileError(
        isDuplicate
          ? "هذا البريد الإلكتروني مستخدم بالفعل. جرّب بريدًا آخر."
          : arabicMessage(
              responseMessage,
              "تعذّر تحديث بيانات الحساب. حاول مرة أخرى.",
            ),
      );
    } finally {
      setProfileLoading(false);
    }
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

  function clearMessages() {
    setCurrentPasswordError("");
    setFormError("");
    setSuccessMessage("");
    setPageError("");
  }

  async function handleChangePassword(event) {
    event.preventDefault();
    clearMessages();

    if (!currentPassword.trim()) {
      setCurrentPasswordError("أدخل كلمة المرور الحالية.");
      return;
    }

    if (!newPassword) {
      setFormError("أدخل كلمة المرور الجديدة.");
      return;
    }

    if (newPassword.length < 8) {
      setFormError("كلمة المرور الجديدة يجب أن تكون ثمانية أحرف على الأقل.");
      return;
    }

    if (newPassword !== confirmPassword) {
      setFormError("كلمتا المرور الجديدتان غير متطابقتين.");
      return;
    }

    if (currentPassword === newPassword) {
      setFormError("اختر كلمة مرور جديدة مختلفة عن الحالية.");
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
      setSuccessMessage("تم تغيير كلمة المرور.");
    } catch (requestError) {
      const responseMessage =
        requestError.response?.data?.message ||
        requestError.response?.data?.error ||
        "";
      const normalized = String(responseMessage).toLowerCase();
      const isCurrentPasswordError =
        normalized.includes("current password") ||
        normalized.includes("incorrect password") ||
        normalized.includes("wrong password") ||
        normalized.includes("password is incorrect") ||
        normalized.includes("invalid current password") ||
        normalized.includes("old password") ||
        responseMessage.includes("كلمة المرور الحالية") ||
        responseMessage.includes("كلمة المرور غير صحيحة") ||
        responseMessage.includes("كلمه المرور الحاليه");

      if (isCurrentPasswordError) {
        setCurrentPasswordError(
          "كلمة المرور الحالية غير صحيحة. حاول مرة أخرى.",
        );
      } else {
        setFormError(
          arabicMessage(
            responseMessage,
            "تعذّر تغيير كلمة المرور. حاول مرة أخرى.",
          ),
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
    } catch {
      setPageError("تعذّر تسجيل الخروج. حاول مرة أخرى.");
    }
  }

  async function handleDeactivateAccount() {
    setPageError("");
    setDeactivating(true);

    try {
      await api.patch("/auth/deactivate-account");
      await logout();
      navigate("/signin", { replace: true });
    } catch (requestError) {
      setPageError(
        arabicMessage(
          requestError.response?.data?.message,
          "تعذّر تعطيل الحساب. حاول مرة أخرى.",
        ),
      );
      setShowDeleteConfirm(false);
    } finally {
      setDeactivating(false);
    }
  }

  return (
    <main className="page account-page">
      <div className="container">
        <header className="account-hero" data-reveal>
          <span className="account-avatar" aria-hidden="true">
            {initials(displayName, "ب")}
          </span>
          <div className="account-hero-copy">
            <p className="eyebrow">حسابك</p>
            <h1>{displayName}</h1>
            <p dir="ltr">{user.email}</p>
          </div>
          <span className="badge badge-accent">{roleLabel}</span>
        </header>

        {pageError ? (
          <div className="alert alert-error" role="alert">
            <i className="bi bi-exclamation-circle" aria-hidden="true"></i>
            <span>{pageError}</span>
          </div>
        ) : null}

        <div className="account-layout">
          <div className="account-main">
            <section className="account-card" aria-labelledby="profile-info-title">
              <div className="account-card-head">
                <span className="account-card-head-icon" aria-hidden="true">
                  <i className="bi bi-person"></i>
                </span>
                <div className="account-card-head-copy">
                  <h2 id="profile-info-title">بيانات الحساب</h2>
                  <p>الاسم والبريد الإلكتروني المرتبطان بحسابك.</p>
                </div>
                {!isEditingProfile ? (
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    onClick={handleOpenProfileEdit}
                  >
                    <i className="bi bi-pencil" aria-hidden="true"></i>
                    تعديل
                  </button>
                ) : null}
              </div>

              {profileSuccess ? (
                <div className="alert alert-success" role="status">
                  <i className="bi bi-check-circle" aria-hidden="true"></i>
                  <span>{profileSuccess}</span>
                </div>
              ) : null}

              {!isEditingProfile ? (
                <dl className="info-list">
                  <div className="info-row">
                    <dt>الاسم</dt>
                    <dd>{displayName}</dd>
                  </div>
                  <div className="info-row">
                    <dt>البريد الإلكتروني</dt>
                    <dd dir="ltr">{user.email || "غير متوفر"}</dd>
                  </div>
                  <div className="info-row">
                    <dt>نوع الحساب</dt>
                    <dd>{roleLabel}</dd>
                  </div>
                </dl>
              ) : (
                <form className="account-form" onSubmit={handleUpdateProfile} noValidate>
                  <div className="field-grid">
                    <div className="field">
                      <label htmlFor="profileFirstName">الاسم الأول</label>
                      <input
                        id="profileFirstName"
                        type="text"
                        value={firstName}
                        onChange={(event) => {
                          setFirstName(event.target.value);
                          setProfileError("");
                        }}
                        placeholder="اسمك الأول"
                        autoComplete="given-name"
                        maxLength={50}
                        disabled={profileLoading}
                        required
                      />
                    </div>

                    <div className="field">
                      <label htmlFor="profileLastName">اسم العائلة</label>
                      <input
                        id="profileLastName"
                        type="text"
                        value={lastName}
                        onChange={(event) => {
                          setLastName(event.target.value);
                          setProfileError("");
                        }}
                        placeholder="اسم العائلة"
                        autoComplete="family-name"
                        maxLength={50}
                        disabled={profileLoading}
                        required
                      />
                    </div>

                    <div className="field field-full">
                      <label htmlFor="profileEmail">البريد الإلكتروني</label>
                      <input
                        id="profileEmail"
                        type="email"
                        value={email}
                        onChange={(event) => {
                          setEmail(event.target.value);
                          setProfileError("");
                        }}
                        placeholder="أدخل بريدك الإلكتروني"
                        autoComplete="email"
                        maxLength={254}
                        dir="ltr"
                        disabled={profileLoading}
                        required
                      />
                    </div>
                  </div>

                  {profileError ? (
                    <div className="alert alert-error" role="alert">
                      <i className="bi bi-exclamation-circle" aria-hidden="true"></i>
                      <span>{profileError}</span>
                    </div>
                  ) : null}

                  <div className="btn-row">
                    <button
                      type="submit"
                      className="btn btn-primary"
                      disabled={profileLoading}
                      aria-busy={profileLoading}
                    >
                      {profileLoading ? (
                        <>
                          <span className="spinner" aria-hidden="true"></span>
                          جارٍ حفظ البيانات…
                        </>
                      ) : (
                        <>
                          <i className="bi bi-check2" aria-hidden="true"></i>
                          حفظ البيانات
                        </>
                      )}
                    </button>

                    <button
                      type="button"
                      className="btn btn-secondary"
                      onClick={handleCancelProfileEdit}
                      disabled={profileLoading}
                    >
                      إلغاء
                    </button>
                  </div>
                </form>
              )}
            </section>

            <section className="account-card" aria-labelledby="profile-password-title">
              <div className="account-card-head">
                <span className="account-card-head-icon" aria-hidden="true">
                  <i className="bi bi-shield-lock"></i>
                </span>
                <div className="account-card-head-copy">
                  <h2 id="profile-password-title">كلمة المرور</h2>
                  <p>استخدم كلمة مرور قوية لا تستخدمها في مواقع أخرى.</p>
                </div>
              </div>

              <form className="account-form" onSubmit={handleChangePassword} noValidate>
                <div className="field">
                  <label htmlFor="currentPassword">كلمة المرور الحالية</label>
                  <div className="password-field">
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
                        currentPasswordError ? "currentPasswordError" : undefined
                      }
                      disabled={loading}
                    />
                    <button
                      type="button"
                      className="password-toggle"
                      onClick={() => setShowCurrentPassword((current) => !current)}
                      aria-label={
                        showCurrentPassword
                          ? "إخفاء كلمة المرور الحالية"
                          : "إظهار كلمة المرور الحالية"
                      }
                      aria-pressed={showCurrentPassword}
                      disabled={loading}
                    >
                      <i
                        className={showCurrentPassword ? "bi bi-eye-slash" : "bi bi-eye"}
                        aria-hidden="true"
                      ></i>
                    </button>
                  </div>
                  {currentPasswordError ? (
                    <span className="field-error" id="currentPasswordError" role="alert">
                      <i className="bi bi-exclamation-circle" aria-hidden="true"></i>
                      {currentPasswordError}
                    </span>
                  ) : null}
                  <div className="auth-meta">
                    <Link to="/forgot-password">نسيت كلمة المرور؟</Link>
                  </div>
                </div>

                <div className="field">
                  <label htmlFor="newPassword">كلمة المرور الجديدة</label>
                  <div className="password-field">
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
                      className="password-toggle"
                      onClick={() => setShowNewPassword((current) => !current)}
                      aria-label={
                        showNewPassword
                          ? "إخفاء كلمة المرور الجديدة"
                          : "إظهار كلمة المرور الجديدة"
                      }
                      aria-pressed={showNewPassword}
                      disabled={loading}
                    >
                      <i
                        className={showNewPassword ? "bi bi-eye-slash" : "bi bi-eye"}
                        aria-hidden="true"
                      ></i>
                    </button>
                  </div>
                  <span className="field-hint">ثمانية أحرف على الأقل.</span>
                </div>

                <div className="field">
                  <label htmlFor="confirmPassword">تأكيد كلمة المرور الجديدة</label>
                  <div className="password-field">
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

                {formError ? (
                  <div className="alert alert-error" role="alert">
                    <i className="bi bi-exclamation-circle" aria-hidden="true"></i>
                    <span>{formError}</span>
                  </div>
                ) : null}

                {successMessage ? (
                  <div className="alert alert-success" role="status">
                    <i className="bi bi-check-circle" aria-hidden="true"></i>
                    <span>{successMessage}</span>
                  </div>
                ) : null}

                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={loading}
                  aria-busy={loading}
                >
                  {loading ? (
                    <>
                      <span className="spinner" aria-hidden="true"></span>
                      جارٍ تغيير كلمة المرور…
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
          </div>

          <aside className="account-side">
            <section className="account-card" aria-labelledby="account-summary-title">
              <div className="account-card-head">
                <span className="account-card-head-icon" aria-hidden="true">
                  <i className="bi bi-person-vcard"></i>
                </span>
                <div className="account-card-head-copy">
                  <h2 id="account-summary-title">ملخص الحساب</h2>
                  <p>هويتك داخل بصيرة.</p>
                </div>
              </div>

              <p><strong>{displayName}</strong></p>
              <p className="small muted" dir="ltr">{user.email}</p>
              <span className="badge badge-accent account-role">{roleLabel}</span>

              <div className="danger-zone">
                <h3>تعطيل الحساب</h3>
                <p>
                  لن تتمكن من استخدام الحساب بعد تعطيله. راجع قرارك قبل
                  المتابعة.
                </p>

                <button
                  type="button"
                  className="btn btn-danger btn-sm"
                  onClick={() => setShowDeleteConfirm(true)}
                >
                  <i className="bi bi-person-x" aria-hidden="true"></i>
                  تعطيل الحساب
                </button>
              </div>
            </section>

            <section className="account-card" aria-labelledby="account-shortcuts-title">
              <div className="account-card-head">
                <span className="account-card-head-icon" aria-hidden="true">
                  <i className="bi bi-grid"></i>
                </span>
                <div className="account-card-head-copy">
                  <h2 id="account-shortcuts-title">روابط سريعة</h2>
                  <p>انتقل مباشرة إلى ما تحتاجه.</p>
                </div>
              </div>

              <ul className="shortcut-list">
                <li>
                  <Link className="shortcut-link" to="/chat">
                    <i className="bi bi-chat-square-text" aria-hidden="true"></i>
                    المساعد المعرفي
                  </Link>
                </li>
                <li>
                  <Link className="shortcut-link" to="/verify">
                    <i className="bi bi-shield-check" aria-hidden="true"></i>
                    التحقق من المعلومات
                  </Link>
                </li>
                <li>
                  <Link className="shortcut-link" to="/services">
                    <i className="bi bi-grid" aria-hidden="true"></i>
                    كل الأدوات
                  </Link>
                </li>
              </ul>

              <button
                type="button"
                className="btn btn-secondary account-signout"
                onClick={handleLogout}
              >
                <i className="bi bi-box-arrow-left" aria-hidden="true"></i>
                تسجيل الخروج
              </button>
            </section>
          </aside>
        </div>
      </div>

      {showDeleteConfirm ? (
        <ConfirmDialog
          title="تعطيل الحساب؟"
          description="لن تتمكن من استخدام حسابك بعد تعطيله، وسيتم تسجيل خروجك. لا يمكن التراجع عن هذا الإجراء بنفسك."
          confirmLabel={deactivating ? "جارٍ التعطيل…" : "تعطيل الحساب"}
          cancelLabel="إلغاء"
          onConfirm={handleDeactivateAccount}
          onClose={() => {
            if (!deactivating) setShowDeleteConfirm(false);
          }}
          busy={deactivating}
          tone="danger"
          icon="bi-person-x"
        />
      ) : null}
    </main>
  );
}

export default Profile;
