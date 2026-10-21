
import { useCallback, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import api from "../services/api";
import { useAuth } from "../context/AuthContext";
import "./AdminDashboard.css";

function AdminDashboard() {
  const { user } = useAuth();

  const [users, setUsers] = useState([]);
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const loadUsers = useCallback(async () => {
    setLoading(true);
    setError("");

    try {
      const response = await api.get("/users");
      const usersData = response.data?.data;

      if (!Array.isArray(usersData)) {
        throw new Error("Unexpected users response");
      }

      setUsers(usersData);
    } catch (err) {
      setError(
        err.response?.data?.message ||
          "تعذر تحميل المستخدمين. تأكد من صلاحيات الأدمن واتصال السيرفر."
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadUsers();
  }, [loadUsers]);

  const filteredUsers = useMemo(() => {
    const query = search.trim().toLowerCase();

    if (!query) return users;

    return users.filter((item) => {
      const fullName =
        `${item.firstName || ""} ${item.lastName || ""}`.toLowerCase();

      return (
        fullName.includes(query) ||
        String(item.email || "").toLowerCase().includes(query)
      );
    });
  }, [users, search]);

  const clearMessages = () => {
    setError("");
    setSuccess("");
  };

  const handleResetUser = async (targetUser) => {
    const fullName =
      `${targetUser.firstName || ""} ${targetUser.lastName || ""}`.trim() ||
      targetUser.email;

    if (
      !window.confirm(
        `هل تريد تصفير الاستخدام الخاص بالمستخدم ${fullName}؟`
      )
    ) {
      return;
    }

    clearMessages();
    setActionLoading(`reset-${targetUser._id}`);

    try {
      await api.post(`/usage/reset/${targetUser._id}`);
      setSuccess(`تم تصفير استخدام ${fullName} بنجاح.`);
    } catch (err) {
      setError(
        err.response?.data?.message || "فشل تصفير استخدام هذا المستخدم."
      );
    } finally {
      setActionLoading("");
    }
  };

  const handleResetAll = async () => {
    if (
      !window.confirm(
        "هل أنت متأكد من تصفير الاستخدام للجميع؟ سيتم حذف سجلات الاستخدام الحالية، بما فيها سجلات الضيوف."
      )
    ) {
      return;
    }

    clearMessages();
    setActionLoading("reset-all");

    try {
      await api.post("/usage/reset-all");
      setSuccess("تم تصفير الاستخدام للجميع بنجاح.");
    } catch (err) {
      setError(
        err.response?.data?.message || "فشل تصفير الاستخدام للجميع."
      );
    } finally {
      setActionLoading("");
    }
  };

  const handleDeleteUser = async (targetUser) => {
    if (String(targetUser._id) === String(user?._id)) {
      setError("لا يمكنك حذف حساب الأدمن الذي تستخدمه حاليًا.");
      return;
    }

    const fullName =
      `${targetUser.firstName || ""} ${targetUser.lastName || ""}`.trim() ||
      targetUser.email;

    if (
      !window.confirm(
        `هل أنت متأكد من حذف المستخدم ${fullName}؟ لا يمكن التراجع عن هذا الإجراء.`
      )
    ) {
      return;
    }

    clearMessages();
    setActionLoading(`delete-${targetUser._id}`);

    try {
      await api.delete(`/users/${targetUser._id}`);

      setUsers((currentUsers) =>
        currentUsers.filter(
          (item) => String(item._id) !== String(targetUser._id)
        )
      );

      setSuccess(`تم حذف المستخدم ${fullName} بنجاح.`);
    } catch (err) {
      setError(
        err.response?.data?.message || "فشل حذف المستخدم."
      );
    } finally {
      setActionLoading("");
    }
  };

  const adminCount = users.filter((item) => item.role === "admin").length;
  const regularUsersCount = users.length - adminCount;

  return (
    <main className="admin-page">
      <div className="admin-container">
        <section className="admin-hero">
          <div className="admin-hero-content">
            <span className="admin-eyebrow">
              <i className="bi bi-shield-check"></i>
              BASIRA AI · ADMINISTRATION
            </span>

            <h1>لوحة تحكم الأدمن</h1>

            <p>
              إدارة المستخدمين ومتابعة الحسابات والتحكم في حدود الاستخدام
              من مكان واحد.
            </p>

            <div className="admin-welcome">
              <span className="admin-avatar">
                <i className="bi bi-person-fill-gear"></i>
              </span>

              <div>
                <span>مرحبًا بك</span>
                <strong>
                  {[user?.firstName, user?.lastName].filter(Boolean).join(" ") ||
                    user?.email ||
                    "Admin"}
                </strong>
              </div>
            </div>
          </div>

          <div className="admin-hero-art" aria-hidden="true">
            <div className="admin-orbit admin-orbit-one"></div>
            <div className="admin-orbit admin-orbit-two"></div>
            <div className="admin-art-icon">
              <i className="bi bi-shield-lock"></i>
            </div>
            <span className="admin-floating-icon admin-float-one">
              <i className="bi bi-people"></i>
            </span>
            <span className="admin-floating-icon admin-float-two">
              <i className="bi bi-activity"></i>
            </span>
          </div>
        </section>

        {error && (
          <div className="admin-alert admin-alert-error" role="alert">
            <i className="bi bi-exclamation-circle-fill"></i>
            <span>{error}</span>
            <button
              type="button"
              aria-label="إغلاق"
              onClick={() => setError("")}
            >
              <i className="bi bi-x-lg"></i>
            </button>
          </div>
        )}

        {success && (
          <div className="admin-alert admin-alert-success" role="status">
            <i className="bi bi-check-circle-fill"></i>
            <span>{success}</span>
            <button
              type="button"
              aria-label="إغلاق"
              onClick={() => setSuccess("")}
            >
              <i className="bi bi-x-lg"></i>
            </button>
          </div>
        )}

        <section className="admin-stats">
          <article className="admin-stat-card">
            <div className="admin-stat-icon stat-icon-purple">
              <i className="bi bi-people-fill"></i>
            </div>
            <div className="admin-stat-info">
              <span>إجمالي المستخدمين</span>
              <strong>{loading ? "—" : users.length}</strong>
              <small>كل الحسابات المسجلة</small>
            </div>
          </article>

          <article className="admin-stat-card">
            <div className="admin-stat-icon stat-icon-blue">
              <i className="bi bi-person-check-fill"></i>
            </div>
            <div className="admin-stat-info">
              <span>المستخدمون العاديون</span>
              <strong>{loading ? "—" : regularUsersCount}</strong>
              <small>حسابات المستخدمين</small>
            </div>
          </article>

          <article className="admin-stat-card">
            <div className="admin-stat-icon stat-icon-green">
              <i className="bi bi-shield-fill-check"></i>
            </div>
            <div className="admin-stat-info">
              <span>الأدمن</span>
              <strong>{loading ? "—" : adminCount}</strong>
              <small>حسابات الإدارة</small>
            </div>
          </article>
        </section>

        <section className="admin-users-panel">
          <div className="admin-panel-heading">
            <div>
              <h2>إدارة المستخدمين</h2>
              <p>عرض الحسابات والبحث بينها وتنفيذ إجراءات الإدارة.</p>
            </div>

            <button
              type="button"
              className="admin-refresh-button"
              onClick={loadUsers}
              disabled={loading || Boolean(actionLoading)}
            >
              <i className="bi bi-arrow-clockwise"></i>
              تحديث البيانات
            </button>
          </div>

          <div className="admin-toolbar">
            <label className="admin-search">
              <i className="bi bi-search"></i>
              <input
                type="search"
                placeholder="ابحث بالاسم او البريد الالكترونى ..."
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                aria-label="البحث عن مستخدم"
              />

              {search && (
                <button
                  type="button"
                  onClick={() => setSearch("")}
                  aria-label="مسح البحث"
                >
                  <i className="bi bi-x-circle"></i>
                </button>
              )}
            </label>

            <button
              type="button"
              className="admin-reset-all-button"
              onClick={handleResetAll}
              disabled={loading || Boolean(actionLoading)}
            >
              {actionLoading === "reset-all" ? (
                <>
                  <span className="admin-spinner"></span>
                  جارٍ التصفير...
                </>
              ) : (
                <>
                  <i className="bi bi-arrow-counterclockwise"></i>
                  تصفير استخدام الجميع
                </>
              )}
            </button>
          </div>

          <div className="admin-table-meta">
            <span>
              <i className="bi bi-list-ul"></i>
              {loading
                ? "جارٍ تحميل البيانات..."
                : `عرض ${filteredUsers.length} من ${users.length} مستخدم`}
            </span>

            <span className="admin-secure-label">
              <i className="bi bi-lock-fill"></i>
              إدارة محمية
            </span>
          </div>

          {loading ? (
            <div className="admin-loading">
              <span className="admin-spinner admin-spinner-large"></span>
              <p>جارٍ تحميل المستخدمين...</p>
            </div>
          ) : error && users.length === 0 ? (
            <div className="admin-empty-state">
              <i className="bi bi-cloud-slash"></i>
              <h3>تعذر تحميل البيانات</h3>
              <p>تأكد من تسجيل الدخول بحساب الأدمن وأن السيرفر يعمل.</p>
              <button
                type="button"
                className="admin-refresh-button"
                onClick={loadUsers}
              >
                إعادة المحاولة
              </button>
            </div>
          ) : filteredUsers.length === 0 ? (
            <div className="admin-empty-state">
              <i className="bi bi-person-exclamation"></i>
              <h3>
                {users.length === 0
                  ? "لا يوجد مستخدمون"
                  : "لا توجد نتائج مطابقة"}
              </h3>
              <p>
                {users.length === 0
                  ? "ستظهر الحسابات هنا عند تسجيل المستخدمين."
                  : "جرّب البحث باستخدام اسم أو بريد إلكتروني مختلف."}
              </p>
            </div>
          ) : (
            <div className="admin-table-wrapper">
              <table className="admin-table">
                <thead>
                  <tr>
                    <th>المستخدم</th>
                    <th>البريد الإلكتروني</th>
                    <th>الدور</th>
                    <th>تاريخ التسجيل</th>
                    <th>الإجراءات</th>
                  </tr>
                </thead>

                <tbody>
                  {filteredUsers.map((item) => {
                    const fullName =
                      `${item.firstName || ""} ${item.lastName || ""}`.trim() ||
                      "مستخدم";

                    const isCurrentUser =
                      String(item._id) === String(user?._id);

                    const rowBusy = actionLoading.endsWith(item._id);

                    return (
                      <tr key={item._id}>
                        <td>
                          <div className="admin-user-cell">
                            <span className="admin-user-avatar">
                              {fullName.charAt(0).toUpperCase()}
                            </span>

                            <div className="admin-user-name">
                              <strong>{fullName}</strong>
                              {isCurrentUser && (
                                <span className="admin-you-badge">
                                  حسابك
                                </span>
                              )}
                            </div>
                          </div>
                        </td>

                        <td>
                          <span className="admin-email">{item.email}</span>
                        </td>

                        <td>
                          <span
                            className={`admin-role-badge ${
                              item.role === "admin"
                                ? "admin-role-admin"
                                : "admin-role-user"
                            }`}
                          >
                            <i
                              className={`bi ${
                                item.role === "admin"
                                  ? "bi-shield-fill-check"
                                  : "bi-person-fill"
                              }`}
                            ></i>
                            {item.role === "admin" ? "أدمن" : "مستخدم"}
                          </span>
                        </td>

                        <td>
                          <span className="admin-date">
                            {item.createdAt
                              ? new Date(item.createdAt).toLocaleDateString(
                                  "ar-EG"
                                )
                              : "—"}
                          </span>
                        </td>

                        <td>
                          <div className="admin-row-actions">
                            <button
                              type="button"
                              className="admin-action-button admin-action-reset"
                              onClick={() => handleResetUser(item)}
                              disabled={Boolean(actionLoading)}
                              title="تصفير استخدام هذا المستخدم"
                            >
                              {actionLoading === `reset-${item._id}` ? (
                                <span className="admin-spinner"></span>
                              ) : (
                                <i className="bi bi-arrow-counterclockwise"></i>
                              )}
                              <span>تصفير الاستخدام</span>
                            </button>

                            <button
                              type="button"
                              className="admin-action-button admin-action-delete"
                              onClick={() => handleDeleteUser(item)}
                              disabled={Boolean(actionLoading) || isCurrentUser}
                              title={
                                isCurrentUser
                                  ? "لا يمكنك حذف حسابك الحالي"
                                  : "حذف المستخدم"
                              }
                            >
                              {actionLoading === `delete-${item._id}` ? (
                                <span className="admin-spinner"></span>
                              ) : (
                                <i className="bi bi-trash3"></i>
                              )}
                              <span>حذف</span>
                            </button>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>
    </main>
  );
}

export default AdminDashboard;
