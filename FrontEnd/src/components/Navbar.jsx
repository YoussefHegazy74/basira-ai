import { useEffect, useRef, useState } from "react";
import { Link, NavLink, useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { useChatNavigation } from "../context/ChatNavigationContext";
import BasiraLogo from "./BasiraLogo";
import ThemeToggle from "./ThemeToggle";
import { formatNumber, initials } from "../utils/format";

const EXPANDED_KEY = "basira.nav.expanded";

const NAV_ITEMS = [
  { to: "/", label: "الرئيسية", icon: "bi-house", end: true },
  {
    to: "/chat",
    label: "المساعد المعرفي",
    icon: "bi-chat-square-text",
  },
  {
    to: "/verify",
    label: "التحقق من المعلومات",
    icon: "bi-shield-check",
  },
  {
    to: "/tafsir",
    label: "البحث في التفسير",
    icon: "bi-book",
  },
  { to: "/services", label: "الخدمات", icon: "bi-grid" },
  { to: "/about", label: "عن بصيرة", icon: "bi-info-circle" },
  {
    to: "/admin",
    label: "لوحة الإدارة",
    icon: "bi-speedometer2",
    admin: true,
  },
];

function conversationTitle(item) {
  return item?.title || "محادثة جديدة";
}

function readExpanded() {
  try {
    const saved = localStorage.getItem(EXPANDED_KEY);
    if (saved === "true") return true;
    if (saved === "false") return false;
  } catch {
    /* storage unavailable */
  }
  return true;
}

function Navbar() {
  const { user, authLoading } = useAuth();
  const {
    conversations,
    activeId,
    loaded,
    busy,
    actions,
  } = useChatNavigation();
  const location = useLocation();
  const navigate = useNavigate();

  const [expanded, setExpanded] = useState(readExpanded);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");

  const railRef = useRef(null);
  const searchRef = useRef(null);
  const menuButtonRef = useRef(null);

  const isChat = location.pathname === "/chat";
  const searchHistory = isChat && Boolean(user);
  const displayName =
    [user?.firstName, user?.lastName].filter(Boolean).join(" ") ||
    user?.name ||
    user?.email ||
    "المستخدم";
  const query = searchQuery.trim().toLowerCase();

  const visibleNavItems = NAV_ITEMS.filter((item) => {
    if (item.admin && user?.role !== "admin") return false;
    if (!searchHistory && query) {
      return item.label.toLowerCase().includes(query);
    }
    return true;
  });

  const visibleConversations = conversations.filter((item) => {
    if (!searchHistory || !query) return true;
    return conversationTitle(item).toLowerCase().includes(query);
  });

  useEffect(() => {
    setMobileOpen(false);
    setSearchQuery("");
  }, [location.pathname]);

  useEffect(() => {
    if (!mobileOpen) return undefined;

    const closeButton = railRef.current?.querySelector(".rail-close");
    closeButton?.focus();

    function handleKeyDown(event) {
      if (event.key === "Escape") {
        setMobileOpen(false);
        menuButtonRef.current?.focus();
        return;
      }

      if (event.key !== "Tab") return;

      const focusable = Array.from(
        railRef.current?.querySelectorAll(
          'a[href], button:not([disabled]), input:not([disabled]), summary'
        ) || []
      ).filter((element) => element.offsetParent !== null);

      if (!focusable.length) return;

      const first = focusable[0];
      const last = focusable[focusable.length - 1];

      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }

    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [mobileOpen]);

  function toggleExpanded() {
    setExpanded((current) => {
      const next = !current;
      try {
        localStorage.setItem(EXPANDED_KEY, String(next));
      } catch {
        /* storage unavailable */
      }
      return next;
    });
  }

  function openMobile() {
    setExpanded(true);
    setMobileOpen(true);
  }

  function closeMobile() {
    const wasOpen = mobileOpen;
    setMobileOpen(false);
    if (wasOpen) menuButtonRef.current?.focus();
  }

  function focusSearch() {
    if (!expanded) {
      setExpanded(true);
      try {
        localStorage.setItem(EXPANDED_KEY, "true");
      } catch {
        /* storage unavailable */
      }
      window.setTimeout(() => searchRef.current?.focus(), 60);
      return;
    }

    searchRef.current?.focus();
  }

  function handleNewChat() {
    setMobileOpen(false);
    navigate("/chat", { state: { newChat: Date.now() } });
  }

  function handleOpenConversation(item) {
    actions.open(item);
    setMobileOpen(false);
    window.setTimeout(() => {
      document.querySelector(".chat-app")?.focus();
    }, 80);
  }

  function handleRename(item) {
    actions.rename(item);
    setMobileOpen(false);
  }

  function handleRemove(item) {
    actions.remove(item);
    setMobileOpen(false);
  }

  return (
    <>
      <header className="mobile-topbar">
        <button
          type="button"
          className="mobile-menu-button"
          ref={menuButtonRef}
          onClick={openMobile}
          aria-label="فتح قائمة التنقل"
          aria-controls="app-rail"
          aria-expanded={mobileOpen}
          title="القائمة"
        >
          <i className="bi bi-list" aria-hidden="true"></i>
        </button>

        <Link
          className="mobile-brand"
          to="/"
          aria-label="بصيرة، الصفحة الرئيسية"
        >
          <BasiraLogo size="md" />
        </Link>

        <span className="mobile-topbar-spacer" aria-hidden="true" />
      </header>

      {mobileOpen ? (
        <button
          type="button"
          className="app-rail-backdrop"
          aria-label="إغلاق قائمة التنقل"
          onClick={closeMobile}
        />
      ) : null}

      <aside
        className={`app-rail ${
          expanded ? "is-expanded" : "is-collapsed"
        } ${mobileOpen ? "is-mobile-open" : ""}`}
        id="app-rail"
        ref={railRef}
        aria-label="التنقل الرئيسي"
      >
        <div className="rail-head">
          <Link
            className="rail-brand"
            to="/"
            aria-label="بصيرة، الصفحة الرئيسية"
            title="بصيرة"
            onClick={closeMobile}
          >
            <BasiraLogo size="md" />
          </Link>

          <button
            type="button"
            className="rail-icon rail-collapse"
            onClick={toggleExpanded}
            aria-label={expanded ? "طي قائمة التنقل" : "توسيع قائمة التنقل"}
            title={expanded ? "طي القائمة" : "توسيع القائمة"}
            aria-expanded={expanded}
          >
            <i
              className={
                expanded
                  ? "bi bi-chevron-double-right"
                  : "bi bi-chevron-double-left"
              }
              aria-hidden="true"
            ></i>
          </button>

          <button
            type="button"
            className="rail-icon rail-close"
            onClick={closeMobile}
            aria-label="إغلاق قائمة التنقل"
            title="إغلاق"
          >
            <i className="bi bi-x-lg" aria-hidden="true"></i>
          </button>
        </div>

        <button
          type="button"
          className="rail-new"
          onClick={handleNewChat}
          disabled={isChat && busy}
          aria-label="بدء محادثة جديدة"
          title="محادثة جديدة"
        >
          <i className="bi bi-plus-lg" aria-hidden="true"></i>
          <span className="rail-label">محادثة جديدة</span>
        </button>

        {expanded ? (
          <label className="rail-search">
            <i className="bi bi-search" aria-hidden="true"></i>
            <span className="sr-only">
              {searchHistory ? "البحث في المحادثات" : "البحث في القائمة"}
            </span>
            <input
              ref={searchRef}
              type="search"
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
              placeholder={
                searchHistory ? "ابحث في المحادثات…" : "ابحث في القائمة…"
              }
              aria-label={
                searchHistory ? "البحث في المحادثات" : "البحث في القائمة"
              }
              autoComplete="off"
            />
            {searchQuery ? (
              <button
                type="button"
                onClick={() => setSearchQuery("")}
                aria-label="مسح البحث"
                title="مسح البحث"
              >
                <i className="bi bi-x-circle" aria-hidden="true"></i>
              </button>
            ) : null}
          </label>
        ) : (
          <button
            type="button"
            className="rail-icon rail-search-trigger"
            onClick={focusSearch}
            aria-label="بحث"
            title="بحث"
          >
            <i className="bi bi-search" aria-hidden="true"></i>
          </button>
        )}

        <div className="rail-body">
          <nav className="rail-nav" aria-label="روابط الموقع">
            {visibleNavItems.map((item) => (
              <NavLink
                className={({ isActive }) =>
                  `rail-nav-item${isActive ? " is-active" : ""}`
                }
                to={item.to}
                key={item.to}
                end={item.end}
                onClick={closeMobile}
                title={item.label}
              >
                <i className={`bi ${item.icon}`} aria-hidden="true"></i>
                <span className="rail-label">{item.label}</span>
              </NavLink>
            ))}
          </nav>

          {!searchHistory && query && visibleNavItems.length === 0 ? (
            <p className="rail-history-empty">لا توجد نتائج مطابقة.</p>
          ) : null}

          {searchHistory ? (
            <section
              className="rail-history-section"
              aria-label="المحادثات الحديثة"
            >
              <div className="rail-section-label">
                <span>المحادثات</span>
                <span>{formatNumber(visibleConversations.length)}</span>
              </div>

              {!loaded ? (
                <div className="rail-history-skeleton" aria-hidden="true">
                  <span className="skeleton"></span>
                  <span className="skeleton"></span>
                  <span className="skeleton"></span>
                </div>
              ) : visibleConversations.length === 0 ? (
                <p className="rail-history-empty">
                  {query
                    ? "لا توجد نتائج مطابقة."
                    : "لا توجد محادثات محفوظة بعد."}
                </p>
              ) : (
                <div className="rail-history">
                  {visibleConversations.map((item) => {
                    const isActive =
                      String(activeId) === String(item._id);

                    return (
                      <div
                        className={`rail-history-row${
                          isActive ? " is-active" : ""
                        }`}
                        key={item._id}
                      >
                        <button
                          type="button"
                          className={`rail-history-item${
                            isActive ? " is-active" : ""
                          }`}
                          onClick={() => handleOpenConversation(item)}
                          disabled={busy}
                          aria-current={isActive ? "true" : undefined}
                          title={conversationTitle(item)}
                        >
                          <i
                            className="bi bi-chat-left-text"
                            aria-hidden="true"
                          ></i>
                          <span>{conversationTitle(item)}</span>
                        </button>

                        <div className="rail-history-actions">
                          <button
                            type="button"
                            className="rail-history-action"
                            onClick={() => handleRename(item)}
                            disabled={busy}
                            aria-label={`تغيير اسم المحادثة: ${conversationTitle(
                              item
                            )}`}
                            title="تغيير الاسم"
                          >
                            <i
                              className="bi bi-pencil-square"
                              aria-hidden="true"
                            ></i>
                          </button>
                          <button
                            type="button"
                            className="rail-history-action danger"
                            onClick={() => handleRemove(item)}
                            disabled={busy}
                            aria-label={`حذف المحادثة: ${conversationTitle(
                              item
                            )}`}
                            title="حذف"
                          >
                            <i
                              className="bi bi-trash3"
                              aria-hidden="true"
                            ></i>
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </section>
          ) : null}
        </div>

        <div className="rail-footer">
          <div className="rail-theme">
            <span className="rail-label">المظهر</span>
            <ThemeToggle />
          </div>

          <Link
            className={`rail-settings${
              location.pathname === "/profile" ? " is-active" : ""
            }`}
            to="/profile"
            aria-label="الإعدادات والحساب"
            title="الإعدادات"
            onClick={closeMobile}
          >
            <i className="bi bi-sliders" aria-hidden="true"></i>
            <span className="rail-label">الإعدادات</span>
          </Link>

          {authLoading ? (
            <span className="rail-user rail-user-loading" aria-hidden="true" />
          ) : user ? (
            <Link
              className="rail-user"
              to="/profile"
              title={displayName}
              onClick={closeMobile}
            >
              <span className="avatar" aria-hidden="true">
                {initials(displayName, "ب")}
              </span>
              <span className="rail-user-copy">
                <strong>{displayName}</strong>
                <span dir="ltr">{user.email}</span>
              </span>
            </Link>
          ) : (
            <Link
              className="rail-signin"
              to="/signin"
              aria-label="تسجيل الدخول"
              title="تسجيل الدخول"
              onClick={closeMobile}
            >
              <i className="bi bi-person" aria-hidden="true"></i>
              <span className="rail-label">تسجيل الدخول</span>
            </Link>
          )}
        </div>
      </aside>
    </>
  );
}

export default Navbar;
