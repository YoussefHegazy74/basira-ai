import { useEffect, useRef, useState } from "react";
import { Link, NavLink, useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import ThemeToggle from "./ThemeToggle";
import { initials } from "../utils/format";

const TOOLS = [
  {
    icon: "bi-chat-square-text",
    title: "المساعد المعرفي",
    description: "اسأل في المسائل الشرعية",
    to: "/chat",
  },
  {
    icon: "bi-shield-check",
    title: "التحقق من المعلومات",
    description: "اعرف مدى دعم النص للمصادر",
    to: "/verify",
  },
  {
    icon: "bi-book",
    title: "البحث في التفسير",
    description: "ابحث عن آية واقرأ تفسيرها",
    to: "/tafsir",
  },
];

function Navbar() {
  const { user, logout, authLoading } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const [openPanel, setOpenPanel] = useState(null);
  const headerRef = useRef(null);
  const toolsRef = useRef(null);
  const profileRef = useRef(null);
  const mobileNavRef = useRef(null);

  const displayName =
    [user?.firstName, user?.lastName].filter(Boolean).join(" ") ||
    user?.name ||
    user?.email ||
    "المستخدم";
  const profileImage =
    user?.profileImage || user?.avatar || user?.photoURL || user?.image || "";
  const toolsActive = ["/chat", "/verify", "/tafsir"].includes(location.pathname);

  useEffect(() => {
    setOpenPanel(null);
  }, [location.pathname]);

  useEffect(() => {
    function handleOutsideClick(event) {
      if (toolsRef.current && !toolsRef.current.contains(event.target)) {
        setOpenPanel((current) => (current === "tools" ? null : current));
      }
      if (profileRef.current && !profileRef.current.contains(event.target)) {
        setOpenPanel((current) => (current === "profile" ? null : current));
      }
      if (
        openPanel === "mobile" &&
        headerRef.current &&
        !headerRef.current.contains(event.target)
      ) {
        setOpenPanel(null);
      }
    }

    function handleEscape(event) {
      if (event.key === "Escape") setOpenPanel(null);
    }

    document.addEventListener("mousedown", handleOutsideClick);
    document.addEventListener("keydown", handleEscape);
    return () => {
      document.removeEventListener("mousedown", handleOutsideClick);
      document.removeEventListener("keydown", handleEscape);
    };
  }, [openPanel]);

  useEffect(() => {
    if (openPanel !== "mobile") return;
    const firstLink = mobileNavRef.current?.querySelector("a, button");
    firstLink?.focus();
  }, [openPanel]);

  function togglePanel(panel) {
    setOpenPanel((current) => (current === panel ? null : panel));
  }

  function closePanels() {
    setOpenPanel(null);
  }

  async function handleLogout() {
    closePanels();
    try {
      await logout();
      navigate("/", { replace: true });
    } catch {
      navigate("/", { replace: true });
    }
  }

  const avatarContent = profileImage ? (
    <img
      className="avatar"
      src={profileImage}
      alt={`صورة ${displayName}`}
    />
  ) : (
    <span className="avatar" aria-hidden="true">
      {initials(displayName, "ب")}
    </span>
  );

  return (
    <header className="site-header" id="top" ref={headerRef}>
      <div className="container header-inner">
        <Link className="brand" to="/" onClick={closePanels} aria-label="بصيرة، الصفحة الرئيسية">
          <span className="brand-mark" aria-hidden="true">
            <i className="bi bi-eye"></i>
          </span>
          <span className="brand-copy">
            <strong>بصيرة</strong>
            <span>معرفة واعية</span>
          </span>
        </Link>

        <nav className="primary-nav" aria-label="التنقل الرئيسي">
          <NavLink to="/" end className="nav-link" onClick={closePanels}>
            الرئيسية
          </NavLink>
          <NavLink to="/services" className="nav-link" onClick={closePanels}>
            الخدمات
          </NavLink>
          <NavLink to="/about" className="nav-link" onClick={closePanels}>
            عن بصيرة
          </NavLink>
          {!authLoading && user?.role === "admin" ? (
            <NavLink to="/admin" className="nav-link" onClick={closePanels}>
              لوحة الإدارة
            </NavLink>
          ) : null}

          <div className="nav-dropdown" ref={toolsRef}>
            <button
              type="button"
              className={`nav-link nav-dropdown-toggle ${toolsActive ? "active" : ""}`}
              onClick={() => togglePanel("tools")}
              aria-expanded={openPanel === "tools"}
              aria-haspopup="true"
              aria-current={toolsActive ? "page" : undefined}
            >
              الأدوات
              <i className="bi bi-chevron-down" aria-hidden="true"></i>
            </button>

            {openPanel === "tools" ? (
              <div className="nav-dropdown-panel">
                <p className="nav-dropdown-header">اختر الأداة المناسبة</p>
                {TOOLS.map((tool) => (
                  <Link
                    className="dropdown-link"
                    to={tool.to}
                    key={tool.to}
                    onClick={closePanels}
                  >
                    <span className="dropdown-link-icon" aria-hidden="true">
                      <i className={`bi ${tool.icon}`}></i>
                    </span>
                    <span className="dropdown-link-copy">
                      <strong>{tool.title}</strong>
                      <span>{tool.description}</span>
                    </span>
                  </Link>
                ))}
              </div>
            ) : null}
          </div>
        </nav>

        <div className="header-actions">
          <ThemeToggle />

          <div className="header-account" ref={profileRef}>
            {!authLoading && user ? (
              <div className="nav-dropdown">
                <button
                  type="button"
                  className="profile-trigger"
                  onClick={() => togglePanel("profile")}
                  aria-expanded={openPanel === "profile"}
                  aria-haspopup="true"
                  aria-label="قائمة الحساب"
                >
                  {avatarContent}
                  <span className="profile-trigger-name">{displayName}</span>
                  <i className="bi bi-chevron-down" aria-hidden="true"></i>
                </button>

                {openPanel === "profile" ? (
                  <div className="profile-panel" role="menu">
                    <div className="profile-panel-head">
                      {avatarContent}
                      <span className="profile-panel-copy">
                        <strong>{displayName}</strong>
                        <span>{user.email}</span>
                      </span>
                    </div>

                    <div className="profile-panel-actions">
                      <Link
                        className="profile-menu-link"
                        to="/profile"
                        role="menuitem"
                        onClick={closePanels}
                      >
                        <i className="bi bi-person" aria-hidden="true"></i>
                        الملف الشخصي
                      </Link>
                      {user.role === "admin" ? (
                        <Link
                          className="profile-menu-link"
                          to="/admin"
                          role="menuitem"
                          onClick={closePanels}
                        >
                          <i className="bi bi-grid" aria-hidden="true"></i>
                          لوحة الإدارة
                        </Link>
                      ) : null}
                      <button
                        type="button"
                        className="profile-menu-button danger"
                        onClick={handleLogout}
                        role="menuitem"
                      >
                        <i className="bi bi-box-arrow-left" aria-hidden="true"></i>
                        تسجيل الخروج
                      </button>
                    </div>
                  </div>
                ) : null}
              </div>
            ) : !authLoading ? (
              <>
                <Link className="header-signin" to="/signin">
                  تسجيل الدخول
                </Link>
                <Link className="btn btn-primary btn-sm" to="/signup">
                  إنشاء حساب
                </Link>
              </>
            ) : null}
          </div>

          <button
            type="button"
            className="menu-toggle"
            onClick={() => togglePanel("mobile")}
            aria-expanded={openPanel === "mobile"}
            aria-controls="mobile-navigation"
            aria-label={openPanel === "mobile" ? "إغلاق القائمة" : "فتح القائمة"}
          >
            <i
              className={openPanel === "mobile" ? "bi bi-x-lg" : "bi bi-list"}
              aria-hidden="true"
            ></i>
          </button>
        </div>
      </div>

      <div
        className={`container mobile-nav ${openPanel === "mobile" ? "is-open" : ""}`}
        id="mobile-navigation"
        ref={mobileNavRef}
      >
        <nav className="mobile-nav-links" aria-label="قائمة الهاتف">
          <NavLink to="/" end className="nav-link" onClick={closePanels}>
            الرئيسية
            <i className="bi bi-arrow-left" aria-hidden="true"></i>
          </NavLink>
          <NavLink to="/services" className="nav-link" onClick={closePanels}>
            الخدمات
            <i className="bi bi-arrow-left" aria-hidden="true"></i>
          </NavLink>
          <NavLink to="/about" className="nav-link" onClick={closePanels}>
            عن بصيرة
            <i className="bi bi-arrow-left" aria-hidden="true"></i>
          </NavLink>
          {!authLoading && user?.role === "admin" ? (
            <NavLink to="/admin" className="nav-link" onClick={closePanels}>
              لوحة الإدارة
              <i className="bi bi-arrow-left" aria-hidden="true"></i>
            </NavLink>
          ) : null}

          <p className="mobile-nav-label">الأدوات</p>
          {TOOLS.map((tool) => (
            <NavLink
              className="nav-link"
              to={tool.to}
              key={tool.to}
              onClick={closePanels}
            >
              {tool.title}
              <i className={`bi ${tool.icon}`} aria-hidden="true"></i>
            </NavLink>
          ))}
        </nav>

        <div className="mobile-nav-account">
          {!authLoading && user ? (
            <>
              <div className="mobile-user">
                {avatarContent}
                <span className="mobile-user-copy">
                  <strong>{displayName}</strong>
                  <span>{user.email}</span>
                </span>
              </div>
              <div className="mobile-user-links">
                <Link
                  className="btn btn-secondary btn-sm"
                  to="/profile"
                  onClick={closePanels}
                >
                  <i className="bi bi-person" aria-hidden="true"></i>
                  الملف الشخصي
                </Link>
                <button
                  type="button"
                  className="btn btn-danger btn-sm"
                  onClick={handleLogout}
                >
                  <i className="bi bi-box-arrow-left" aria-hidden="true"></i>
                  تسجيل الخروج
                </button>
              </div>
            </>
          ) : !authLoading ? (
            <>
              <Link
                className="btn btn-secondary"
                to="/signin"
                onClick={closePanels}
              >
                تسجيل الدخول
              </Link>
              <Link
                className="btn btn-primary"
                to="/signup"
                onClick={closePanels}
              >
                إنشاء حساب
              </Link>
            </>
          ) : null}
        </div>
      </div>
    </header>
  );
}

export default Navbar;
