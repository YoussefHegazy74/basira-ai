import { useEffect, useState } from "react";
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
import ChatSidebar from "./components/ChatSidebar";
import Footer from "./components/Footer";
import ErrorBoundary from "./components/ErrorBoundary";
import Home from "./pages/Home";
import Chat from "./pages/Chat";
import Verify from "./pages/Verify";
import Tafsir from "./pages/Tafsir";
import About from "./pages/About";
import Services from "./pages/Services";
import { AuthProvider, useAuth } from "./context/AuthContext";
import { ChatNavigationProvider } from "./context/ChatNavigationContext";
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
    const main = document.getElementById("main-content");
    if (main) main.scrollTo({ top: 0, left: 0, behavior: "auto" });
    window.scrollTo({ top: 0, left: 0, behavior: "auto" });
  }, [pathname]);

  useEffect(() => {
    const reduced = window.matchMedia?.(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    const container = document.getElementById("main-content") || document.body;

    function revealAll() {
      container
        .querySelectorAll("[data-reveal]")
        .forEach((element) => element.classList.add("is-visible"));
    }

    if (reduced || !("IntersectionObserver" in window)) {
      revealAll();
      const mutation = new MutationObserver(revealAll);
      mutation.observe(container, { childList: true, subtree: true });
      return () => mutation.disconnect();
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

    function observePending() {
      container
        .querySelectorAll("[data-reveal]:not(.is-visible)")
        .forEach((element) => observer.observe(element));
    }

    observePending();
    const mutation = new MutationObserver(observePending);
    mutation.observe(container, { childList: true, subtree: true });

    return () => {
      observer.disconnect();
      mutation.disconnect();
    };
  }, [pathname]);

  return null;
}

const CHAT_SIDEBAR_KEY = "basira.chat-sidebar";

function readChatSidebarCollapsed() {
  if (typeof window === "undefined") return true;

  try {
    const stored = localStorage.getItem(CHAT_SIDEBAR_KEY);
    if (stored === "collapsed") return true;
    if (stored === "expanded") return false;
  } catch {
    /* storage unavailable */
  }

  return !window.matchMedia?.("(min-width: 1024px)")?.matches;
}

function AppLayout() {
  const { pathname } = useLocation();
  const isChat = pathname === "/chat";
  const [chatSidebarCollapsed, setChatSidebarCollapsed] = useState(
    readChatSidebarCollapsed
  );
  const [chatDrawerOpen, setChatDrawerOpen] = useState(false);

  usePageMeta();

  useEffect(() => {
    setChatDrawerOpen(false);
  }, [pathname]);

  useEffect(() => {
    if (!isChat) return;
    try {
      localStorage.setItem(
        CHAT_SIDEBAR_KEY,
        chatSidebarCollapsed ? "collapsed" : "expanded"
      );
    } catch {
      /* storage unavailable */
    }
  }, [chatSidebarCollapsed, isChat]);

  function closeChatDrawer() {
    setChatDrawerOpen(false);
    window.requestAnimationFrame(() => {
      document.getElementById("chat-sidebar-open")?.focus();
    });
  }

  return (
    <>
      <a className="skip-link" href="#main-content">
        {isChat ? "انتقل إلى المحادثة" : "انتقل إلى المحتوى"}
      </a>
      <ScrollAndReveal />
      <div className="app-shell">
        <Navbar />
        <div className={`app-body${isChat ? " app-body-chat" : ""}`}>
          {isChat ? (
            <ChatSidebar
              collapsed={chatSidebarCollapsed}
              mobileOpen={chatDrawerOpen}
              onToggleCollapsed={() =>
                setChatSidebarCollapsed((current) => !current)
              }
              onCloseMobile={() => setChatDrawerOpen(false)}
            />
          ) : null}

          {isChat && chatDrawerOpen ? (
            <button
              type="button"
              className="chat-sidebar-backdrop"
              aria-label="إغلاق قائمة المحادثات"
              onClick={closeChatDrawer}
            />
          ) : null}

          <div
            className={`app-main${isChat ? " app-main-chat" : ""}`}
            id="main-content"
            tabIndex={-1}
          >
            <div className="app-content">
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
                <Route
                  path="/chat"
                  element={
                    <Chat
                      chatSidebarOpen={chatDrawerOpen}
                      onOpenChatSidebar={() => setChatDrawerOpen(true)}
                    />
                  }
                />
                <Route path="/verify" element={<Verify />} />
                <Route path="/tafsir" element={<Tafsir />} />
                <Route
                  path="/auth/resetpassword/:token"
                  element={<ResetPassword />}
                />
                <Route path="*" element={<NotFound />} />
              </Routes>
              {!isChat ? <Footer /> : null}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

function App() {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <AuthProvider>
          <ChatNavigationProvider>
            <AppLayout />
          </ChatNavigationProvider>
        </AuthProvider>
      </BrowserRouter>
    </ErrorBoundary>
  );
}

export default App;
