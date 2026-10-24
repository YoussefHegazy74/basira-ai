import { useCallback, useEffect, useMemo, useState } from "react";
import api from "../services/api";
import { useAuth } from "../context/AuthContext";
import { ConfirmDialog } from "../components/Modal";
import {
  arabicMessage,
  formatCount,
  formatDate,
  initials,
} from "../utils/format";
import "../styles/admin.css";

const ROLE_FILTERS = [
  { id: "all", label: "الكل" },
  { id: "admin", label: "الإدارة" },
  { id: "user", label: "المستخدمون" },
];

function userName(item) {
  return (
    `${item.firstName || ""} ${item.lastName || ""}`.trim() ||
    item.name ||
    item.email ||
    "مستخدم"
  );
}

function AdminDashboard() {
  const { user } = useAuth();

  const [users, setUsers] = useState([]);
  const [search, setSearch] = useState("");
  const [roleFilter, setRoleFilter] = useState("all");
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [dialog, setDialog] = useState(null);

  const loadUsers = useCallback(async () => {
    setLoading(true);
    setError("");

    try {
      const response = await api.get("/users");
      const usersData = response.data?.data;

      if (!Array.isArray(usersData)) {
        throw new Error("تعذّر قراءة بيانات المستخدمين.");
      }

      setUsers(usersData);
    } catch (requestError) {
      setError(
        arabicMessage(
          requestError.response?.data?.message,
          "تعذّر تحميل المستخدمين. تأكد من صلاحيات الإدارة واتصال الخدمة."
        )
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadUsers();
  }, [loadUsers]);

  const adminCount = users.filter((item) => item.role === "admin").length;
  const regularUsersCount = users.length - adminCount;

  const filteredUsers = useMemo(() => {
    const query = search.trim().toLowerCase();

    return users.filter((item) => {
      if (roleFilter === "admin" && item.role !== "admin") return false;
      if (roleFilter === "user" && item.role === "admin") return false;
      if (!query) return true;

      const fullName = `${item.firstName || ""} ${
        item.lastName || ""
      }`.toLowerCase();

      return (
        fullName.includes(query) ||
        String(item.email || "").toLowerCase().includes(query)
      );
    });
  }, [users, search, roleFilter]);

  const currentUserId = user?._id || user?.id;
  const displayName =
    [user?.firstName, user?.lastName].filter(Boolean).join(" ") ||
    user?.name ||
    user?.email ||
    "مدير النظام";

  function openResetUser(targetUser) {
    setError("");
    setSuccess("");
    setDialog({ type: "reset", user: targetUser });
  }

  function openDeleteUser(targetUser) {
    if (String(targetUser._id) === String(currentUserId)) {
      setError("لا يمكنك حذف الحساب الذي تستخدمه حاليًا.");
      return;
    }

    setError("");
    setSuccess("");
    setDialog({ type: "delete", user: targetUser });
  }

  function openResetAll() {
    setError("");
    setSuccess("");
    setDialog({ type: "reset-all" });
  }

  async function handleDialogConfirm() {
    if (!dialog || actionLoading) return;

    if (dialog.type === "reset-all") {
      setActionLoading("reset-all");

      try {
        await api.post("/usage/reset-all");
        setSuccess("تم تصفير استخدام الجميع.");
      } catch (requestError) {
        setError(
          arabicMessage(
            requestError.response?.data?.message,
            "تعذّر تصفير استخدام الجميع. حاول مرة أخرى."
          )
        );
      } finally {
        setActionLoading("");
        setDialog(null);
      }

      return;
    }

    const targetUser = dialog.user;
    const fullName = userName(targetUser);

    if (dialog.type === "reset") {
      setActionLoading(`reset-${targetUser._id}`);

      try {
        await api.post(`/usage/reset/${targetUser._id}`);
        setSuccess(`تم تصفير استخدام ${fullName}.`);
      } catch (requestError) {
        setError(
          arabicMessage(
            requestError.response?.data?.message,
            "تعذّر تصفير استخدام هذا المستخدم. حاول مرة أخرى."
          )
        );
      } finally {
        setActionLoading("");
        setDialog(null);
      }

      return;
    }

    setActionLoading(`delete-${targetUser._id}`);

    try {
      await api.delete(`/users/${targetUser._id}`);

      setUsers((currentUsers) =>
        currentUsers.filter(
          (item) => String(item._id) !== String(targetUser._id)
        )
      );

      setSuccess(`تم حذف حساب ${fullName}.`);
    } catch (requestError) {
      setError(
        arabicMessage(
          requestError.response?.data?.message,
          "تعذّر حذف المستخدم. حاول مرة أخرى."
        )
      );
    } finally {
      setActionLoading("");
      setDialog(null);
    }
  }

  const dialogBusy = Boolean(actionLoading);
  const dialogUserName = dialog?.user ? userName(dialog.user) : "";

  return (
    <main className="page admin-page">
      <div className="container admin-shell">
        <header className="admin-head" data-reveal>
          <div>
            <p className="eyebrow">الإدارة</p>
            <h1>لوحة الإدارة</h1>
            <p>
              راجع حسابات المستخدمين، وابحث بينها، وأدر عدّادات الاستخدام
              من مكان واحد.
            </p>
          </div>

          <div className="admin-welcome">
            <span className="admin-welcome-avatar" aria-hidden="true">
              {initials(displayName, "م")}
            </span>
            <span className="admin-welcome-copy">
              <span>مرحبًا بك</span>
              <strong>{displayName}</strong>
            </span>
          </div>
        </header>

        {error ? (
          <div className="alert alert-error admin-alert" role="alert">
            <i className="bi bi-exclamation-circle" aria-hidden="true"></i>
            <span>{error}</span>
            <button
              type="button"
              className="admin-alert-close"
              onClick={() => setError("")}
              aria-label="إغلاق التنبيه"
              title="إغلاق"
            >
              <i className="bi bi-x-lg" aria-hidden="true"></i>
            </button>
          </div>
        ) : null}

        {success ? (
          <div className="alert alert-success admin-alert" role="status">
            <i className="bi bi-check-circle" aria-hidden="true"></i>
            <span>{success}</span>
            <button
              type="button"
              className="admin-alert-close"
              onClick={() => setSuccess("")}
              aria-label="إغلاق التنبيه"
              title="إغلاق"
            >
              <i className="bi bi-x-lg" aria-hidden="true"></i>
            </button>
          </div>
        ) : null}

        <section className="admin-stats" aria-label="ملخص الحسابات">
          <article className="admin-stat">
            <span className="admin-stat-icon" aria-hidden="true">
              <i className="bi bi-people"></i>
            </span>
            <div>
              <span className="admin-stat-label">إجمالي المستخدمين</span>
              {loading ? (
                <span
                  className="skeleton admin-stat-skeleton"
                  aria-hidden="true"
                ></span>
              ) : (
                <strong className="admin-stat-value">
                  {formatCount(users.length)}
                </strong>
              )}
              <small>كل الحسابات المسجلة</small>
            </div>
          </article>

          <article className="admin-stat">
            <span className="admin-stat-icon" aria-hidden="true">
              <i className="bi bi-person-check"></i>
            </span>
            <div>
              <span className="admin-stat-label">المستخدمون</span>
              {loading ? (
                <span
                  className="skeleton admin-stat-skeleton"
                  aria-hidden="true"
                ></span>
              ) : (
                <strong className="admin-stat-value">
                  {formatCount(regularUsersCount)}
                </strong>
              )}
              <small>حسابات بلا صلاحيات إدارة</small>
            </div>
          </article>

          <article className="admin-stat">
            <span className="admin-stat-icon" aria-hidden="true">
              <i className="bi bi-shield-check"></i>
            </span>
            <div>
              <span className="admin-stat-label">الإدارة</span>
              {loading ? (
                <span
                  className="skeleton admin-stat-skeleton"
                  aria-hidden="true"
                ></span>
              ) : (
                <strong className="admin-stat-value">
                  {formatCount(adminCount)}
                </strong>
              )}
              <small>حسابات بصلاحيات إدارة</small>
            </div>
          </article>
        </section>

        <section className="admin-panel" aria-labelledby="admin-users-title">
          <div className="admin-panel-head">
            <div>
              <h2 id="admin-users-title">المستخدمون</h2>
              <p>ابحث عن حساب، أو راجع الاستخدام، أو احذف حسابًا.</p>
            </div>

            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={loadUsers}
              disabled={loading || Boolean(actionLoading)}
            >
              {loading ? (
                <>
                  <span className="spinner" aria-hidden="true"></span>
                  جارٍ التحديث…
                </>
              ) : (
                <>
                  <i
                    className="bi bi-arrow-clockwise"
                    aria-hidden="true"
                  ></i>
                  تحديث البيانات
                </>
              )}
            </button>
          </div>

          <div className="admin-toolbar">
            <label className="admin-search">
              <span className="sr-only">البحث عن مستخدم</span>
              <i className="bi bi-search" aria-hidden="true"></i>
              <input
                type="search"
                placeholder="ابحث بالاسم أو البريد الإلكتروني…"
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                aria-label="البحث عن مستخدم"
              />

              {search ? (
                <button
                  type="button"
                  onClick={() => setSearch("")}
                  aria-label="مسح البحث"
                  title="مسح البحث"
                >
                  <i className="bi bi-x-circle" aria-hidden="true"></i>
                </button>
              ) : null}
            </label>

            <div
              className="admin-filters"
              role="group"
              aria-label="تصفية المستخدمين حسب الدور"
            >
              {ROLE_FILTERS.map((filter) => (
                <button
                  type="button"
                  key={filter.id}
                  className={roleFilter === filter.id ? "is-active" : ""}
                  onClick={() => setRoleFilter(filter.id)}
                  aria-pressed={roleFilter === filter.id}
                >
                  {filter.label}
                  <span>
                    {formatCount(
                      filter.id === "all"
                        ? users.length
                        : filter.id === "admin"
                          ? adminCount
                          : regularUsersCount
                    )}
                  </span>
                </button>
              ))}
            </div>

            <button
              type="button"
              className="btn btn-danger btn-sm admin-reset-all"
              onClick={openResetAll}
              disabled={loading || Boolean(actionLoading)}
            >
              <i
                className="bi bi-arrow-counterclockwise"
                aria-hidden="true"
              ></i>
              تصفير استخدام الجميع
            </button>
          </div>

          <div className="admin-meta">
            <span>
              <i className="bi bi-list-ul" aria-hidden="true"></i>
              {loading
                ? "جارٍ تحميل البيانات…"
                : `عرض ${formatCount(filteredUsers.length)} من ${formatCount(
                    users.length
                  )} حساب`}
            </span>

            <span className="admin-secure-label">
              <i className="bi bi-lock" aria-hidden="true"></i>
              إدارة محمية
            </span>
          </div>

          {loading ? (
            <div
              className="admin-table-skeleton"
              aria-live="polite"
              aria-busy="true"
            >
              <span className="sr-only">جارٍ تحميل المستخدمين…</span>
              <div aria-hidden="true">
                <span className="skeleton"></span>
                <span className="skeleton"></span>
                <span className="skeleton"></span>
                <span className="skeleton"></span>
                <span className="skeleton"></span>
              </div>
            </div>
          ) : error && users.length === 0 ? (
            <div className="admin-empty">
              <span className="admin-empty-icon" aria-hidden="true">
                <i className="bi bi-cloud-slash"></i>
              </span>
              <h3>تعذّر تحميل البيانات</h3>
              <p>تأكد من تسجيل الدخول بحساب إداري وأن الخدمة تعمل.</p>
              <button
                type="button"
                className="btn btn-primary btn-sm"
                onClick={loadUsers}
              >
                إعادة المحاولة
              </button>
            </div>
          ) : filteredUsers.length === 0 ? (
            <div className="admin-empty">
              <span className="admin-empty-icon" aria-hidden="true">
                <i className="bi bi-person-exclamation"></i>
              </span>
              <h3>
                {users.length === 0
                  ? "لا يوجد مستخدمون بعد"
                  : "لا توجد نتائج مطابقة"}
              </h3>
              <p>
                {users.length === 0
                  ? "ستظهر الحسابات هنا عند تسجيل المستخدمين."
                  : "جرّب اسمًا أو بريدًا إلكترونيًا مختلفًا، أو غيّر التصفية."}
              </p>
              {search || roleFilter !== "all" ? (
                <button
                  type="button"
                  className="btn btn-secondary btn-sm"
                  onClick={() => {
                    setSearch("");
                    setRoleFilter("all");
                  }}
                >
                  مسح البحث والتصفية
                </button>
              ) : null}
            </div>
          ) : (
            <div className="admin-table-wrap">
              <table className="admin-table">
                <caption className="sr-only">
                  قائمة حسابات المستخدمين
                </caption>
                <thead>
                  <tr>
                    <th scope="col">المستخدم</th>
                    <th scope="col">البريد الإلكتروني</th>
                    <th scope="col">الدور</th>
                    <th scope="col">تاريخ التسجيل</th>
                    <th scope="col">الإجراءات</th>
                  </tr>
                </thead>

                <tbody>
                  {filteredUsers.map((item) => {
                    const fullName = userName(item);
                    const isCurrentUser =
                      String(item._id) === String(currentUserId);
                    const isResetting =
                      actionLoading === `reset-${item._id}`;
                    const isDeleting =
                      actionLoading === `delete-${item._id}`;
                    const roleLabel =
                      item.role === "admin" ? "إدارة" : "مستخدم";

                    return (
                      <tr key={item._id}>
                        <td data-label="المستخدم">
                          <div className="admin-user-cell">
                            <span
                              className="admin-user-avatar"
                              aria-hidden="true"
                            >
                              {initials(fullName, "م")}
                            </span>

                            <span className="admin-user-name">
                              <strong>{fullName}</strong>
                              {isCurrentUser ? (
                                <span className="admin-self">حسابك</span>
                              ) : null}
                            </span>
                          </div>
                        </td>

                        <td data-label="البريد الإلكتروني">
                          <span className="admin-email" dir="ltr">
                            {item.email || "غير متوفر"}
                          </span>
                        </td>

                        <td data-label="الدور">
                          <span
                            className={`admin-role ${
                              item.role === "admin" ? "is-admin" : ""
                            }`}
                          >
                            <i
                              className={`bi ${
                                item.role === "admin"
                                  ? "bi-shield-check"
                                  : "bi-person"
                              }`}
                              aria-hidden="true"
                            ></i>
                            {roleLabel}
                          </span>
                        </td>

                        <td data-label="تاريخ التسجيل">
                          {item.createdAt ? (
                            <time
                              className="admin-date"
                              dateTime={item.createdAt}
                            >
                              {formatDate(item.createdAt)}
                            </time>
                          ) : (
                            <span className="muted">—</span>
                          )}
                        </td>

                        <td data-label="الإجراءات">
                          <div className="admin-row-actions">
                            <button
                              type="button"
                              className="btn btn-secondary btn-sm"
                              onClick={() => openResetUser(item)}
                              disabled={Boolean(actionLoading)}
                              title="تصفير استخدام هذا المستخدم"
                              aria-label={`تصفير استخدام ${fullName}`}
                            >
                              {isResetting ? (
                                <span
                                  className="spinner"
                                  aria-hidden="true"
                                ></span>
                              ) : (
                                <i
                                  className="bi bi-arrow-counterclockwise"
                                  aria-hidden="true"
                                ></i>
                              )}
                              <span>تصفير الاستخدام</span>
                            </button>

                            <button
                              type="button"
                              className="btn btn-danger btn-sm"
                              onClick={() => openDeleteUser(item)}
                              disabled={
                                Boolean(actionLoading) || isCurrentUser
                              }
                              title={
                                isCurrentUser
                                  ? "لا يمكنك حذف حسابك الحالي"
                                  : `حذف حساب ${fullName}`
                              }
                              aria-label={
                                isCurrentUser
                                  ? "لا يمكنك حذف حسابك الحالي"
                                  : `حذف حساب ${fullName}`
                              }
                            >
                              {isDeleting ? (
                                <span
                                  className="spinner"
                                  aria-hidden="true"
                                ></span>
                              ) : (
                                <i
                                  className="bi bi-trash3"
                                  aria-hidden="true"
                                ></i>
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

      {dialog ? (
        <ConfirmDialog
          title={
            dialog.type === "reset-all"
              ? "تصفير استخدام الجميع؟"
              : dialog.type === "reset"
                ? "تصفير استخدام المستخدم؟"
                : "حذف المستخدم؟"
          }
          description={
            dialog.type === "reset-all"
              ? "ستُحذف سجلات الاستخدام الحالية لجميع الحسابات، بما فيها سجلات الزوار. لا يمكن التراجع عن هذا الإجراء."
              : dialog.type === "reset"
                ? `سيُعاد عدّاد استخدام «${dialogUserName}» إلى الصفر. لا يؤثر ذلك على الحساب أو محادثاته.`
                : `سيتم حذف حساب «${dialogUserName}». لا يمكن التراجع عن هذا الإجراء.`
          }
          confirmLabel={
            dialog.type === "reset-all"
              ? "تصفير الجميع"
              : dialog.type === "reset"
                ? "تصفير الاستخدام"
                : "حذف الحساب"
          }
          onConfirm={handleDialogConfirm}
          onClose={() => {
            if (!dialogBusy) setDialog(null);
          }}
          busy={dialogBusy}
          tone={
            dialog.type === "reset-all" || dialog.type === "delete"
              ? "danger"
              : ""
          }
          icon={
            dialog.type === "delete"
              ? "bi-trash3"
              : dialog.type === "reset-all"
                ? "bi-exclamation-triangle"
                : "bi-arrow-counterclockwise"
          }
        />
      ) : null}
    </main>
  );
}

export default AdminDashboard;
