
import { useEffect, useRef, useState } from "react";
import { NavLink, Link, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

function Navbar() {
  const { user, logout, authLoading } = useAuth();
  const navigate = useNavigate();

  const [isBotsMenuOpen, setIsBotsMenuOpen] = useState(false);
  const [isProfileMenuOpen, setIsProfileMenuOpen] = useState(false);

  const botsMenuRef = useRef(null);
  const profileMenuRef = useRef(null);

  const bots = [
    {
      icon: "bi-chat-square-text",
      title: "المساعد الذكي",
      description: "اسأل واحصل على إجابات",
      link: "/chat",
    },
    {
      icon: "bi-shield-check",
      title: "التحقق من المعلومات",
      description: "تحقق من موثوقية المعلومات",
      link: "/verify",
    },
    {
      icon: "bi-book-half",
      title: "التفسير",
      description: "استكشف خدمة التفسير",
      link: "/tafsir",
    },
  ];

  const displayName = user
    ? `${user.firstName || ""} ${user.lastName || ""}`.trim() ||
      user.name ||
      user.email ||
      "المستخدم"
    : "";

  const userEmail = user?.email || "";

  // صورة المستخدم لو موجودة في بيانات الحساب
  const profileImage =
    user?.profileImage ||
    user?.avatar ||
    user?.photoURL ||
    user?.image ||
    "";

  // إغلاق القوائم عند الضغط خارجها أو الضغط على Escape
  useEffect(function () {
    function handleOutsideClick(event) {
      if (
        botsMenuRef.current &&
        !botsMenuRef.current.contains(event.target)
      ) {
        setIsBotsMenuOpen(false);
      }

      if (
        profileMenuRef.current &&
        !profileMenuRef.current.contains(event.target)
      ) {
        setIsProfileMenuOpen(false);
      }
    }

    function handleEscape(event) {
      if (event.key === "Escape") {
        setIsBotsMenuOpen(false);
        setIsProfileMenuOpen(false);
      }
    }

    document.addEventListener("mousedown", handleOutsideClick);
    document.addEventListener("keydown", handleEscape);

    return function () {
      document.removeEventListener("mousedown", handleOutsideClick);
      document.removeEventListener("keydown", handleEscape);
    };
  }, []);

  function closeBotsMenu() {
    setIsBotsMenuOpen(false);
  }

  function closeProfileMenu() {
    setIsProfileMenuOpen(false);
  }

  function toggleBotsMenu() {
    setIsBotsMenuOpen(function (previousValue) {
      return !previousValue;
    });

    setIsProfileMenuOpen(false);
  }

  function toggleProfileMenu() {
    setIsProfileMenuOpen(function (previousValue) {
      return !previousValue;
    });

    setIsBotsMenuOpen(false);
  }

  async function handleLogout() {
    closeBotsMenu();
    closeProfileMenu();

    try {
      await logout();
      navigate("/", { replace: true });
    } catch (error) {
      console.error("Logout error:", error);
    }
  }

  function openProfile() {
    closeBotsMenu();
    closeProfileMenu();
    navigate("/profile");
  }

  return (
    <header className="navbar">
      <div className="container nav-content">
        {/* Logo */}
        <Link
          className="brand"
          to="/"
          onClick={function () {
            closeBotsMenu();
            closeProfileMenu();
          }}
        >
          <span className="brand-icon">
            <i className="bi bi-eye-fill"></i>
          </span>

          <span>
            بصيرة <small>Basira AI</small>
          </span>
        </Link>

        {/* Main Navigation */}
        <nav className="nav-links">
          <NavLink
            to="/"
            end
            className="nav-link"
            onClick={function () {
              closeBotsMenu();
              closeProfileMenu();
            }}
          >
            الرئيسية
          </NavLink>

          <NavLink
            to="/services"
            className="nav-link"
            onClick={function () {
              closeBotsMenu();
              closeProfileMenu();
            }}
          >
            خدماتنا
          </NavLink>

          <NavLink
            to="/about"
            className="nav-link"
            onClick={function () {
              closeBotsMenu();
              closeProfileMenu();
            }}
          >
            عن بصيرة
          </NavLink>

          {!authLoading && user?.role === "admin" && (
            <NavLink
              to="/admin"
              className="nav-link"
              onClick={function () {
                closeBotsMenu();
                closeProfileMenu();
              }}
            >
              <i className="bi bi-speedometer2"></i>{" "}
              لوحة الإدارة
            </NavLink>
          )}
        </nav>

        {/* Auth + Dropdown Menus */}
        <div className="nav-auth">
          {/* Bots Dropdown */}
          <div className="nav-bots-wrapper" ref={botsMenuRef}>
            <button
              type="button"
              className={`nav-bots-toggle ${
                isBotsMenuOpen ? "active" : ""
              }`}
              onClick={toggleBotsMenu}
              aria-label="قائمة البوتات"
              aria-expanded={isBotsMenuOpen}
              aria-haspopup="true"
              title="خدمات بصيرة"
            >
              <i
                className={
                  isBotsMenuOpen
                    ? "bi bi-x-lg"
                    : "bi bi-grid-3x3-gap-fill"
                }
              ></i>
            </button>

            {isBotsMenuOpen && (
              <div className="nav-bots-dropdown">
                <div className="bots-dropdown-header">
                  <span className="bots-header-icon">
                    <i className="bi bi-stars"></i>
                  </span>

                  <div>
                    <h3>خدمات بصيرة</h3>
                    <p>اختر الخدمة التي تحتاجها</p>
                  </div>
                </div>

                <div className="bots-dropdown-list">
                  {bots.map(function (bot) {
                    return (
                      <NavLink
                        to={bot.link}
                        key={bot.link}
                        className="bot-dropdown-item"
                        onClick={closeBotsMenu}
                      >
                        <span className="bot-item-icon">
                          <i className={`bi ${bot.icon}`}></i>
                        </span>

                        <span className="bot-item-info">
                          <strong>{bot.title}</strong>
                          <small>{bot.description}</small>
                        </span>

                        <i className="bi bi-arrow-up-left bot-item-arrow"></i>
                      </NavLink>
                    );
                  })}
                </div>

                <Link
                  to="/services"
                  className="bots-dropdown-footer"
                  onClick={closeBotsMenu}
                >
                  عرض كل الخدمات
                  <i className="bi bi-arrow-left"></i>
                </Link>
              </div>
            )}
          </div>

          {/* Profile Dropdown */}
          {!authLoading && user ? (
            <div className="nav-profile-wrapper" ref={profileMenuRef}>
              <button
                type="button"
                className={`nav-profile-toggle ${
                  isProfileMenuOpen ? "active" : ""
                }`}
                onClick={toggleProfileMenu}
                aria-label="قائمة الحساب"
                aria-expanded={isProfileMenuOpen}
                aria-haspopup="true"
                title="حسابي"
              >
                {profileImage ? (
                  <img
                    src={profileImage}
                    alt="الصورة الشخصية"
                    className="nav-profile-avatar"
                  />
                ) : (
                  <i className="bi bi-person-circle"></i>
                )}

                <span className="nav-profile-toggle-name">
                  {displayName}
                </span>

                <i
                  className={`bi ${
                    isProfileMenuOpen
                      ? "bi-chevron-up"
                      : "bi-chevron-down"
                  } nav-profile-chevron`}
                ></i>
              </button>

              {isProfileMenuOpen && (
                <div className="nav-profile-dropdown">
                  {/* User Information */}
                  <button
                    type="button"
                    className="profile-dropdown-user"
                    onClick={openProfile}
                  >
                    <span className="profile-dropdown-avatar">
                      {profileImage ? (
                        <img
                          src={profileImage}
                          alt="الصورة الشخصية"
                        />
                      ) : (
                        <i className="bi bi-person-fill"></i>
                      )}
                    </span>

                    <span className="profile-dropdown-info">
                      <strong>{displayName}</strong>
                      <small>{userEmail}</small>

                      <span className="profile-dropdown-view">
                        عرض الملف الشخصي
                        <i className="bi bi-arrow-up-left"></i>
                      </span>
                    </span>
                  </button>

                  {/* Divider */}
                  <div className="profile-dropdown-divider"></div>

                  {/* Logout */}
                  <button
                    type="button"
                    className="profile-dropdown-logout"
                    onClick={handleLogout}
                  >
                    <span className="profile-logout-icon">
                      <i className="bi bi-box-arrow-right"></i>
                    </span>

                    <span>تسجيل الخروج</span>
                  </button>
                </div>
              )}
            </div>
          ) : !authLoading ? (
            <>
              <Link
                className="nav-signin"
                to="/signin"
                onClick={function () {
                  closeBotsMenu();
                  closeProfileMenu();
                }}
              >
                تسجيل الدخول
              </Link>

              <Link
                className="nav-button"
                to="/signup"
                onClick={function () {
                  closeBotsMenu();
                  closeProfileMenu();
                }}
              >
                إنشاء حساب <i className="bi bi-arrow-left"></i>
              </Link>
            </>
          ) : null}
        </div>
      </div>
    </header>
  );
}

export default Navbar;
