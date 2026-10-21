
import { NavLink, Link, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

function Navbar() {
  const { user, logout, authLoading } = useAuth();
  const navigate = useNavigate();

  async function handleLogout() {
    await logout();
    navigate("/", { replace: true });
  }

  const displayName = user
    ? `${user.firstName || ""} ${user.lastName || ""}`.trim() ||
      user.email
    : "";

  return (
    <header className="navbar">
      <div className="container nav-content">
        <Link className="brand" to="/">
          <span className="brand-icon">
            <i className="bi bi-eye-fill"></i>
          </span>

          <span>
            بصيرة <small>Basira AI</small>
          </span>
        </Link>

        <nav className="nav-links">
          <NavLink to="/" end className="nav-link">
            الرئيسية
          </NavLink>

          <NavLink to="/services" className="nav-link">
            خدماتنا
          </NavLink>

          <NavLink to="/about" className="nav-link">
            عن بصيرة
          </NavLink>

          {/* يظهر رابط الإدارة للأدمن فقط */}
          {!authLoading && user?.role === "admin" && (
            <NavLink to="/admin" className="nav-link">
              <i className="bi bi-speedometer2"></i>{" "}
              لوحة الإدارة
            </NavLink>
          )}
        </nav>

        <div className="nav-auth">
          {!authLoading && user ? (
            <>
              <span className="nav-user-name" title={user.email}>
                <i className="bi bi-person-circle"></i>
                <span>{displayName}</span>
              </span>

              <button
                type="button"
                className="nav-logout"
                onClick={handleLogout}
              >
                تسجيل الخروج{" "}
                <i className="bi bi-box-arrow-right"></i>
              </button>
            </>
          ) : !authLoading ? (
            <>
              <Link className="nav-signin" to="/signin">
                تسجيل الدخول
              </Link>

              <Link className="nav-button" to="/signup">
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
