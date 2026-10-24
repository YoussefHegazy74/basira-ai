import { useEffect, useState } from "react";

const THEME_KEY = "basira.theme";
const MEDIA_QUERY = "(prefers-color-scheme: dark)";

function readTheme() {
  try {
    const stored = localStorage.getItem(THEME_KEY);
    if (stored === "light" || stored === "dark") return stored;
  } catch {
    /* storage unavailable */
  }
  if (typeof window !== "undefined" && window.matchMedia) {
    return window.matchMedia(MEDIA_QUERY).matches ? "dark" : "light";
  }
  return "light";
}

function applyTheme(theme, animate = false) {
  const root = document.documentElement;
  if (animate) {
    root.classList.add("theme-transition");
    window.setTimeout(() => root.classList.remove("theme-transition"), 260);
  }
  root.dataset.theme = theme;
  root.style.colorScheme = theme;
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) {
    meta.setAttribute("content", theme === "dark" ? "#151a17" : "#f7f5ef");
  }
}

function ThemeToggle() {
  const [theme, setTheme] = useState(readTheme);

  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  useEffect(() => {
    if (!window.matchMedia) return undefined;
    const media = window.matchMedia(MEDIA_QUERY);

    function handleChange(event) {
      let stored = null;
      try {
        stored = localStorage.getItem(THEME_KEY);
      } catch {
        /* storage unavailable */
      }
      if (!stored) {
        const next = event.matches ? "dark" : "light";
        setTheme(next);
        applyTheme(next, true);
      }
    }

    media.addEventListener("change", handleChange);
    return () => media.removeEventListener("change", handleChange);
  }, []);

  function toggleTheme() {
    const next = theme === "dark" ? "light" : "dark";
    try {
      localStorage.setItem(THEME_KEY, next);
    } catch {
      /* storage unavailable */
    }
    setTheme(next);
    applyTheme(next, true);
  }

  const isDark = theme === "dark";
  const label = isDark ? "تفعيل الوضع النهاري" : "تفعيل الوضع الليلي";

  return (
    <button
      type="button"
      className="theme-toggle"
      onClick={toggleTheme}
      aria-label={label}
      title={label}
      aria-pressed={isDark}
    >
      <i className={isDark ? "bi bi-sun" : "bi bi-moon"} aria-hidden="true"></i>
    </button>
  );
}

export default ThemeToggle;
