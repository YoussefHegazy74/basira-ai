import { useEffect } from "react";
import {
  BrowserRouter,
  Link,
  Navigate,
  Route,
  Routes,
  useLocation,
} from "react-router-dom";

import Profile from "./pages/Profile";
import ForgotPassword from "./pages/ForgotPassword";
import ResetPassword from "./pages/ResetPassword";
import AdminDashboard from "./pages/AdminDashboard";
import SignIn from "./pages/SignIn";
import SignUp from "./pages/SignUp";
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";
import ErrorBoundary from "./components/ErrorBoundary";
import Home from "./pages/Home";
import Chat from "./pages/Chat";
import Verify from "./pages/Verify";
import Tafsir from "./pages/Tafsir";
import About from "./pages/About";
import Services from "./pages/Services";
import { AuthProvider, useAuth } from "./context/AuthContext";
import { usePageMeta } from "./hooks/usePageMeta";
import { formatNumber } from "./utils/format";

function NotFound() {
  return (
    <main className="page not-found-page">
      <div className="container not-found-inner">
        <div className="not-found-code" aria-hidden="true">
          {formatNumber(404)}
        </div>
        <p className="eyebrow">صفحة غير موجودة</p>
        <h1>يبدو أن هذا الرابط لا يقود إلى مكان</h1>
        <p>
          ربما تغيّر الرابط أو كُتب بشكل غير مكتمل. يمكنك العودة إلى الصفحة
          الرئيسية، أو الانتقال مباشرة إلى إحدى الأدوات.
        </p>
        <div className="not-found-actions">
          <Link className="btn btn-primary" to="/">
            العودة إلى الرئيسية
          </Link>
          <Link className="btn btn-secondary" to="/services">
            استكشاف الأدوات
          </Link>
        </div>
      </div>
    </main>
  );
}

function AdminRoute({ children }) {
  const { user, authLoading } = useAuth();

  if (authLoading) {
    return (
      <div className="route-state" aria-live="polite">
        <span className="spinner" aria-hidden="true"></span>
        <p>جارٍ التحقق من الصلاحيات…</p>
      </div>
    );
  }

  if (!user) return <Navigate to="/signin" replace />;
  if (user.role !== "admin") return <Navigate to="/" replace />;
  return children;
}

function GuestRoute({ children }) {
  const { user, authLoading } = useAuth();

  if (authLoading) {
    return (
      <div className="route-state" aria-live="polite">
        <span className="spinner" aria-hidden="true"></span>
        <p>جارٍ التحقق من حالة الدخول…</p>
      </div>
    );
  }

  if (user) return <Navigate to="/" replace />;
  return children;
}

function ScrollAndReveal() {
  const { pathname } = useLocation();

  useEffect(() => {
    window.scrollTo({ top: 0, left: 0, behavior: "auto" });
  }, [pathname]);

  useEffect(() => {
    const elements = Array.from(document.querySelectorAll("[data-reveal]"));
    if (!elements.length) return undefined;

    const reduced = window.matchMedia?.(
      "(prefers-reduced-motion: reduce)",
    ).matches;

    if (reduced || !("IntersectionObserver" in window)) {
      elements.forEach((element) => element.classList.add("is-visible"));
      return undefined;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            observer.unobserve(entry.target);
          }
        });
      },
      { rootMargin: "0px 0px -6% 0px", threshold: 0.08 },
    );

    elements.forEach((element) => observer.observe(element));
    return () => observer.disconnect();
  }, [pathname]);

  return null;
}

function AppLayout() {
  usePageMeta();

  return (
    <>
      <a className="skip-link" href="#main-content">
        انتقل إلى المحتوى
      </a>
      <ScrollAndReveal />
      <Navbar />
      <div className="app-content" id="main-content" tabIndex={-1}>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/about" element={<About />} />
          <Route path="/services" element={<Services />} />

          <Route
            path="/signin"
            element={
              <GuestRoute>
                <SignIn />
              </GuestRoute>
            }
          />
          <Route
            path="/signup"
            element={
              <GuestRoute>
                <SignUp />
              </GuestRoute>
            }
          />
          <Route path="/forgot-password" element={<ForgotPassword />} />

          <Route
            path="/admin"
            element={
              <AdminRoute>
                <AdminDashboard />
              </AdminRoute>
            }
          />

          <Route path="/profile" element={<Profile />} />
          <Route path="/chat" element={<Chat />} />
          <Route path="/verify" element={<Verify />} />
          <Route path="/tafsir" element={<Tafsir />} />
          <Route
            path="/auth/resetpassword/:token"
            element={<ResetPassword />}
          />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </div>
      <Footer />
    </>
  );
}

function App() {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <AuthProvider>
          <AppLayout />
        </AuthProvider>
      </BrowserRouter>
    </ErrorBoundary>
  );
}

export default App;
