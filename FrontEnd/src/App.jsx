
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";

import Profile from "./pages/Profile";
import ForgotPassword from "./pages/ForgotPassword";
import ResetPassword from "./pages/ResetPassword";
import AdminDashboard from "./pages/AdminDashboard";

import SignIn from "./pages/SignIn";
import SignUp from "./pages/SignUp";

import Navbar from "./components/Navbar";
import Footer from "./components/Footer";

import Home from "./pages/Home";
import Chat from "./pages/Chat";
import Verify from "./pages/Verify";
import Tafsir from "./pages/Tafsir";
import About from "./pages/About";
import Services from "./pages/Services";

import "./App.css";
import { AuthProvider, useAuth } from "./context/AuthContext";

function NotFound() {
  return (
    <main className="container" style={{ paddingBlock: "100px" }}>
      <h1>الصفحة غير موجودة</h1>
      <p>تأكد من الرابط وحاول مرة أخرى.</p>
    </main>
  );
}

// حماية صفحة الأدمن
function AdminRoute({ children }) {
  const { user, authLoading } = useAuth();

  if (authLoading) {
    return (
      <div className="admin-route-loading">
        <span className="admin-spinner admin-spinner-large"></span>
        <p>جارٍ التحقق من الصلاحيات...</p>
      </div>
    );
  }

  if (!user) {
    return <Navigate to="/signin" replace />;
  }

  if (user.role !== "admin") {
    return <Navigate to="/" replace />;
  }

  return children;
}

// منع المستخدم المسجل دخوله من فتح تسجيل الدخول أو إنشاء حساب
function GuestRoute({ children }) {
  const { user, authLoading } = useAuth();

  if (authLoading) {
    return (
      <div className="admin-route-loading">
        <span className="admin-spinner admin-spinner-large"></span>
        <p>جارٍ التحقق من حالة تسجيل الدخول...</p>
      </div>
    );
  }

  if (user) {
    return <Navigate to="/" replace />;
  }

  return children;
}

function AppLayout() {
  return (
    <>
      <Navbar />

      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/about" element={<About />} />
        <Route path="/services" element={<Services />} />

        {/* صفحات تسجيل الدخول وإنشاء الحساب */}
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

        <Route
          path="/forgot-password"
          element={<ForgotPassword />}
        />

        {/* صفحة الأدمن */}
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

      <Footer />
    </>
  );
}

function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <AppLayout />
      </AuthProvider>
    </BrowserRouter>
  );
}

export default App;
