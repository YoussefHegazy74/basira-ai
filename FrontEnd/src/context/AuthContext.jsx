import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
} from "react";
import api from "../services/api";

const AuthContext = createContext(null);
const SESSION_KEY = "basira.session";

function readSession() {
  try {
    const value = localStorage.getItem(SESSION_KEY);
    return value ? JSON.parse(value) : null;
  } catch {
    return null;
  }
}

function normalizeUser(user) {
  if (!user) return null;
  return {
    ...user,
    email: String(user.email || "").trim().toLowerCase(),
    role: user.role === "admin" ? "admin" : "user",
  };
}

export function AuthProvider({ children }) {
  const [user, setUserState] = useState(() => normalizeUser(readSession()));
  const [authLoading, setAuthLoading] = useState(true);

  const setUser = useCallback((nextUser) => {
    const normalized = normalizeUser(nextUser);
    setUserState(normalized);
    try {
      if (normalized) localStorage.setItem(SESSION_KEY, JSON.stringify(normalized));
      else localStorage.removeItem(SESSION_KEY);
    } catch (error) {
      console.warn("Could not persist the local session:", error);
    }
  }, []);

  const refreshUser = useCallback(async () => {
    const savedUser = normalizeUser(readSession());
    if (!savedUser) {
      setUserState(null);
      setAuthLoading(false);
      return null;
    }

    try {
      // The supplied backend does not expose /auth/me. /usage is a real
      // protected endpoint and can confirm whether the session cookie works.
      await api.get("/usage");
      if (savedUser.role === "admin") {
        // Admin-only endpoint verifies admin privileges on the server.
        await api.get("/users");
      }
      setUserState(savedUser);
      return savedUser;
    } catch {
      setUserState(null);
      try { localStorage.removeItem(SESSION_KEY); } catch { /* storage unavailable */ }
      return null;
    } finally {
      setAuthLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshUser();
  }, [refreshUser]);

  const logout = useCallback(async () => {
    try {
      // The backend protects logout, so send it before clearing local state.
      await api.post("/auth/logout");
    } catch (error) {
      console.warn(
        "Logout request failed:",
        error.response?.data?.message || error.message
      );
    } finally {
      setUser(null);
      setAuthLoading(false);
    }
  }, [setUser]);

  return (
    <AuthContext.Provider value={{ user, setUser, refreshUser, logout, authLoading }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth must be used inside AuthProvider");
  return context;
}
