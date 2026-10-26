import { useEffect, useRef, useState } from "react";
import { Link, NavLink, useLocation } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import BasiraLogo from "./BasiraLogo";
import ThemeToggle from "./ThemeToggle";
import { initials } from "../utils/format";

const SERVICES = [
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
];

function isServicePath(pathname) {
  return SERVICES.some((service) => pathname === service.to);
}

function Navbar() {
  const { user, authLoading } = useAuth();
  const location = useLocation();

  const [mobileOpen, setMobileOpen] = useState(false);
  const [mobileServicesOpen, setMobileServicesOpen] = useState(false);
  const [servicesOpen, setServicesOpen] = useState(false);
  const [servicesVisible, setServicesVisible] = useState(false);
  const [servicesPosition, setServicesPosition] = useState(null);

  const menuButtonRef = useRef(null);
  const panelRef = useRef(null);
  const servicesWrapperRef = useRef(null);
  const servicesTriggerRef = useRef(null);
  const servicesMenuRef = useRef(null);
  const servicesCloseTimerRef = useRef(null);
  const servicesOpenIntentRef = useRef(false);

  const isServicesRoute = isServicePath(location.pathname);
  const displayName =
    [user?.firstName, user?.lastName].filter(Boolean).join(" ") ||
    user?.name ||
    user?.email ||
    "المستخدم";

  const adminItem =
    user?.role === "admin"
      ? { to: "/admin", label: "لوحة الإدارة" }
      : null;

  function focusServicesTrigger() {
    servicesTriggerRef.current?.focus();
  }

  function positionServices() {
    const trigger = servicesTriggerRef.current;
    if (!trigger) return;

    const rect = trigger.getBoundingClientRect();
    const header = trigger.closest(".site-header");
    const headerBottom =
      header?.getBoundingClientRect().bottom ?? rect.bottom;
    const width = Math.min(600, window.innerWidth - 24);
    const centered = rect.left + rect.width / 2 - width / 2;
    const left = Math.min(
      Math.max(12, centered),
      Math.max(12, window.innerWidth - width - 12)
    );

    setServicesPosition({
      left: Math.round(left),
      top: Math.round(headerBottom + 10),
      width: Math.round(width),
    });
  }

  function openServices() {
    servicesOpenIntentRef.current = true;

    if (servicesCloseTimerRef.current) {
      window.clearTimeout(servicesCloseTimerRef.current);
      servicesCloseTimerRef.current = null;
    }

    positionServices();

    if (!servicesVisible) {
      setServicesOpen(false);
      setServicesVisible(true);
      return;
    }

    setServicesOpen(true);
  }

  function closeServices({ restoreFocus = false } = {}) {
    servicesOpenIntentRef.current = false;

    if (servicesCloseTimerRef.current) {
      window.clearTimeout(servicesCloseTimerRef.current);
    }

    setServicesOpen(false);
    servicesCloseTimerRef.current = window.setTimeout(() => {
      setServicesVisible(false);
      servicesCloseTimerRef.current = null;
    }, 180);

    if (restoreFocus) focusServicesTrigger();
  }

  function toggleServices() {
    if (servicesOpenIntentRef.current) closeServices({ restoreFocus: true });
    else openServices();
  }

  function handleServicesKeyDown(event) {
    if (event.key === "Escape") {
      event.preventDefault();
      closeServices({ restoreFocus: true });
      return;
    }

    if (event.key === "Tab") {
      event.preventDefault();
      closeServices({ restoreFocus: true });
      return;
    }

    const items = Array.from(
      servicesMenuRef.current?.querySelectorAll('[role="menuitem"]') || []
    );
    if (!items.length) return;

    const currentIndex = items.indexOf(document.activeElement);
    const rtl = document.documentElement.dir === "rtl";
    let nextIndex = null;

    if (event.key === "Home") nextIndex = 0;
    else if (event.key === "End") nextIndex = items.length - 1;
    else if (event.key === "ArrowDown") nextIndex = currentIndex + 1;
    else if (event.key === "ArrowUp") nextIndex = currentIndex - 1;
    else if (event.key === "ArrowLeft") nextIndex = currentIndex + (rtl ? 1 : -1);
    else if (event.key === "ArrowRight") nextIndex = currentIndex + (rtl ? -1 : 1);

    if (nextIndex === null) return;

    event.preventDefault();
    const bounded = (nextIndex + items.length) % items.length;
    items[bounded].focus();
  }

  useEffect(() => {
    setMobileOpen(false);
    setMobileServicesOpen(false);
    if (servicesVisible) closeServices();
    // Only react to route changes; menu close is handled by the functions above.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [location.pathname]);

  useEffect(() => {
    if (!servicesVisible) return undefined;

    const frame = window.requestAnimationFrame(() => {
      if (servicesOpenIntentRef.current) setServicesOpen(true);
    });
    return () => window.cancelAnimationFrame(frame);
  }, [servicesVisible]);

  useEffect(() => {
    if (!servicesOpen) return undefined;

    const frame = window.requestAnimationFrame(() => {
      servicesMenuRef.current
        ?.querySelector('[role="menuitem"]')
        ?.focus({ preventScroll: true });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [servicesOpen]);

  useEffect(() => {
    if (!servicesVisible) return undefined;

    function handlePointerDown(event) {
      if (!servicesWrapperRef.current?.contains(event.target)) {
        closeServices();
      }
    }

    function handleResize() {
      if (window.innerWidth <= 767) {
        closeServices();
        return;
      }
      positionServices();
    }

    document.addEventListener("pointerdown", handlePointerDown, true);
    window.addEventListener("resize", handleResize);
    return () => {
      document.removeEventListener("pointerdown", handlePointerDown, true);
      window.removeEventListener("resize", handleResize);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [servicesVisible]);

  useEffect(() => {
    return () => {
      if (servicesCloseTimerRef.current) {
        window.clearTimeout(servicesCloseTimerRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (!mobileOpen) return undefined;

    const firstLink = panelRef.current?.querySelector("a, button");
    firstLink?.focus();

    function handleKeyDown(event) {
      if (event.key === "Escape") {
        setMobileOpen(false);
        menuButtonRef.current?.focus();
      }
    }

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [mobileOpen]);

  return (
    <>
      <header className="site-header">
        <div className="site-header-inner">
          <Link
            className="site-brand"
            to="/"
            aria-label="بصيرة، الصفحة الرئيسية"
          >
            <BasiraLogo size="md" />
          </Link>

          <nav className="site-nav" aria-label="التنقل الرئيسي">
            <NavLink
              className={({ isActive }) =>
                `site-nav-link${isActive ? " is-active" : ""}`
              }
              to="/"
              end
            >
              الرئيسية
            </NavLink>

            <div className="site-services" ref={servicesWrapperRef}>
              <button
                type="button"
                ref={servicesTriggerRef}
                className={`site-nav-link site-services-trigger${
                  isServicesRoute ? " is-active" : ""
                }${servicesOpen ? " is-open" : ""}`}
                onClick={toggleServices}
                onKeyDown={(event) => {
                  if (event.key === "ArrowDown") {
                    event.preventDefault();
                    openServices();
                  }
                }}
                aria-haspopup="menu"
                aria-expanded={servicesOpen}
                aria-controls="site-services-menu"
              >
                <span>الخدمات</span>
                <i
                  className="bi bi-chevron-down site-services-chevron"
                  aria-hidden="true"
                ></i>
              </button>

              {servicesVisible ? (
                <div
                  className={`services-capsule${
                    servicesOpen ? " is-open" : ""
                  }`}
                  id="site-services-menu"
                  ref={servicesMenuRef}
                  role="menu"
                  aria-label="الخدمات"
                  aria-hidden={servicesOpen ? undefined : "true"}
                  inert={servicesOpen ? undefined : true}
                  style={
                    servicesPosition
                      ? {
                          left: `${servicesPosition.left}px`,
                          top: `${servicesPosition.top}px`,
                          width: `${servicesPosition.width}px`,
                        }
                      : undefined
                  }
                  onKeyDown={handleServicesKeyDown}
                >
                  {SERVICES.map((service, index) => (
                    <NavLink
                      key={service.to}
                      className={({ isActive }) =>
                        `services-capsule-item${
                          isActive ? " is-active" : ""
                        }`
                      }
                      to={service.to}
                      role="menuitem"
                      style={{ "--service-index": index }}
                      onClick={() => closeServices()}
                    >
                      <i
                        className={`bi ${service.icon}`}
                        aria-hidden="true"
                      ></i>
                      <span>{service.label}</span>
                    </NavLink>
                  ))}
                </div>
              ) : null}
            </div>

            <NavLink
              className={({ isActive }) =>
                `site-nav-link${isActive ? " is-active" : ""}`
              }
              to="/about"
            >
              عن بصيرة
            </NavLink>

            {adminItem ? (
              <NavLink
                className={({ isActive }) =>
                  `site-nav-link${isActive ? " is-active" : ""}`
                }
                to={adminItem.to}
              >
                {adminItem.label}
              </NavLink>
            ) : null}
          </nav>

          <div className="site-actions">
            <ThemeToggle />

            {authLoading ? (
              <span className="site-user-loading" aria-hidden="true" />
            ) : user ? (
              <Link
                className="site-user"
                to="/profile"
                title={displayName}
                aria-label={`الحساب: ${displayName}`}
              >
                <span className="avatar" aria-hidden="true">
                  {initials(displayName, "ب")}
                </span>
                <span className="site-user-name">{displayName}</span>
              </Link>
            ) : (
              <Link className="site-signin" to="/signin">
                <i className="bi bi-person" aria-hidden="true"></i>
                <span>تسجيل الدخول</span>
              </Link>
            )}

            <button
              type="button"
              className="site-menu-toggle"
              ref={menuButtonRef}
              onClick={() => {
                setMobileOpen((current) => !current);
                closeServices();
              }}
              aria-label={
                mobileOpen ? "إغلاق قائمة التنقل" : "فتح قائمة التنقل"
              }
              aria-controls="site-mobile-nav"
              aria-expanded={mobileOpen}
              title="القائمة"
            >
              <i
                className={mobileOpen ? "bi bi-x-lg" : "bi bi-list"}
                aria-hidden="true"
              ></i>
            </button>
          </div>
        </div>

        {mobileOpen ? (
          <nav
            className="site-mobile-nav"
            id="site-mobile-nav"
            ref={panelRef}
            aria-label="روابط الموقع"
          >
            <NavLink
              className={({ isActive }) =>
                `site-mobile-link${isActive ? " is-active" : ""}`
              }
              to="/"
              end
              onClick={() => setMobileOpen(false)}
            >
              الرئيسية
              <i className="bi bi-chevron-left" aria-hidden="true"></i>
            </NavLink>

            <div className="site-mobile-services">
              <button
                type="button"
                className={`site-mobile-link site-mobile-services-trigger${
                  isServicesRoute ? " is-active" : ""
                }${mobileServicesOpen ? " is-open" : ""}`}
                onClick={() => setMobileServicesOpen((current) => !current)}
                aria-expanded={mobileServicesOpen}
                aria-controls="site-mobile-services-panel"
              >
                <span>الخدمات</span>
                <i
                  className="bi bi-chevron-down site-mobile-chevron"
                  aria-hidden="true"
                ></i>
              </button>

              <div
                className={`site-mobile-services-panel${
                  mobileServicesOpen ? " is-open" : ""
                }`}
                id="site-mobile-services-panel"
                aria-hidden={mobileServicesOpen ? undefined : "true"}
              >
                <div
                  className="site-mobile-services-inner"
                  inert={mobileServicesOpen ? undefined : true}
                >
                  {SERVICES.map((service) => (
                    <NavLink
                      key={service.to}
                      className={({ isActive }) =>
                        `site-mobile-service-link${
                          isActive ? " is-active" : ""
                        }`
                      }
                      to={service.to}
                      onClick={() => setMobileOpen(false)}
                    >
                      <i
                        className={`bi ${service.icon}`}
                        aria-hidden="true"
                      ></i>
                      <span>{service.label}</span>
                    </NavLink>
                  ))}
                </div>
              </div>
            </div>

            <NavLink
              className={({ isActive }) =>
                `site-mobile-link${isActive ? " is-active" : ""}`
              }
              to="/about"
              onClick={() => setMobileOpen(false)}
            >
              عن بصيرة
              <i className="bi bi-chevron-left" aria-hidden="true"></i>
            </NavLink>

            {adminItem ? (
              <NavLink
                className={({ isActive }) =>
                  `site-mobile-link${isActive ? " is-active" : ""}`
                }
                to={adminItem.to}
                onClick={() => setMobileOpen(false)}
              >
                {adminItem.label}
                <i className="bi bi-chevron-left" aria-hidden="true"></i>
              </NavLink>
            ) : null}
          </nav>
        ) : null}
      </header>

      {mobileOpen ? (
        <button
          type="button"
          className="site-nav-backdrop"
          aria-label="إغلاق قائمة التنقل"
          onClick={() => {
            setMobileOpen(false);
            menuButtonRef.current?.focus();
          }}
        />
      ) : null}
    </>
  );
}

export default Navbar;
