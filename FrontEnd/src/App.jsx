
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Profile from "./pages/Profile";

import ForgotPassword from "./pages/ForgotPassword";
import ResetPassword from "./pages/ResetPassword";

import AdminDashboard from "./pages/AdminDashboard";
import { Navigate } from "react-router-dom";
import { useAuth } from "./context/AuthContext";

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
import { AuthProvider } from "./context/AuthContext";

function NotFound() {
  return (
    <main className="container" style={{ paddingBlock: "100px" }}>
      <h1>الصفحة غير موجودة</h1>
      <p>تأكد من الرابط وحاول مرة أخرى.</p>
    </main>
  );
}
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

function AppLayout() {
  return (
    <>
      <Navbar />

      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/about" element={<About />} />
        <Route path="/services" element={<Services />} />

        <Route path="/signin" element={<SignIn />} />
        <Route path="/signup" element={<SignUp />} />

        <Route
          path="/forgot-password"
          element={<ForgotPassword />}
        />
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

        {/* Admin Dashboard */}

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
