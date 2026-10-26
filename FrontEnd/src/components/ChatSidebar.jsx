import { useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { useChatNavigation } from "../context/ChatNavigationContext";
import ThemeToggle from "./ThemeToggle";
import { formatNumber, initials } from "../utils/format";

const MOBILE_QUERY = "(max-width: 767px)";

function conversationTitle(item) {
  return item?.title || "محادثة جديدة";
}

function isNarrowScreen() {
  return (
    typeof window !== "undefined" &&
    window.matchMedia?.(MOBILE_QUERY).matches
  );
}

function ChatSidebar({ collapsed, mobileOpen, onToggleCollapsed, onCloseMobile }) {
  const { user, authLoading } = useAuth();
  const {
    conversations,
    activeId,
    loaded,
    busy,
    actions,
  } = useChatNavigation();
  const navigate = useNavigate();

  const [isMobile, setIsMobile] = useState(isNarrowScreen);
  const [searchQuery, setSearchQuery] = useState("");
  const searchRef = useRef(null);
  const sidebarRef = useRef(null);
  const bodyRef = useRef(null);
  const focusTimerRef = useRef(null);

  const railCollapsed = Boolean(collapsed) && !isMobile;
  const toggleLabel = isMobile
    ? "إغلاق قائمة المحادثات"
    : railCollapsed
      ? "توسيع قائمة المحادثات"
      : "طي قائمة المحادثات";

  const query = searchQuery.trim().toLowerCase();
  const visibleConversations = conversations.filter((item) => {
    if (!query) return true;
    return conversationTitle(item).toLowerCase().includes(query);
  });

  const displayName =
    [user?.firstName, user?.lastName].filter(Boolean).join(" ") ||
    user?.name ||
    user?.email ||
    "المستخدم";

  useEffect(() => {
    const media = window.matchMedia(MOBILE_QUERY);

    function handleChange(event) {
      setIsMobile(event.matches);
    }

    media.addEventListener("change", handleChange);
    return () => media.removeEventListener("change", handleChange);
  }, []);

  useEffect(() => {
    return () => {
      if (focusTimerRef.current) window.clearTimeout(focusTimerRef.current);
    };
  }, []);

  useEffect(() => {
    if (!mobileOpen || !isMobile) return undefined;

    const focusTarget = searchRef.current || sidebarRef.current;
    const timer = window.setTimeout(() => focusTarget?.focus(), 40);

    function returnFocus() {
      window.requestAnimationFrame(() => {
        document.getElementById("chat-sidebar-open")?.focus();
      });
    }

    function handleKeyDown(event) {
      if (event.key === "Escape") {
        onCloseMobile?.();
        returnFocus();
        return;
      }
      if (event.key !== "Tab") return;

      const focusable = Array.from(
        sidebarRef.current?.querySelectorAll(
          'a[href], button:not([disabled]), input:not([disabled])'
        ) || []
      ).filter((element) => element.offsetParent !== null);

      if (!focusable.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];

      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }

    document.addEventListener("keydown", handleKeyDown);
    return () => {
      window.clearTimeout(timer);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [mobileOpen, isMobile, onCloseMobile]);

  function closeOnNarrow() {
    if (isNarrowScreen()) onCloseMobile?.();
  }

  function handleToggle() {
    if (isMobile) {
      onCloseMobile?.();
      window.requestAnimationFrame(() => {
        document.getElementById("chat-sidebar-open")?.focus();
      });
      return;
    }
    onToggleCollapsed?.();
  }

  function handleNewChat() {
    actions.newChat();
    closeOnNarrow();
  }

  function handleSearchActivate() {
    if (!railCollapsed) {
      searchRef.current?.focus();
      return;
    }

    onToggleCollapsed?.();
    if (focusTimerRef.current) window.clearTimeout(focusTimerRef.current);
    focusTimerRef.current = window.setTimeout(() => {
      searchRef.current?.focus();
    }, 300);
  }

  function handleHistoryActivate() {
    if (railCollapsed) onToggleCollapsed?.();
    if (focusTimerRef.current) window.clearTimeout(focusTimerRef.current);
    focusTimerRef.current = window.setTimeout(() => {
      bodyRef.current?.scrollTo({ top: 0, behavior: "smooth" });
    }, railCollapsed ? 300 : 0);
  }

  function handleOpenConversation(item) {
    actions.open(item);
    closeOnNarrow();
    window.setTimeout(() => {
      document.querySelector(".chat-app")?.focus();
    }, 80);
  }

  function handleRename(item) {
    actions.rename(item);
    closeOnNarrow();
  }

  function handleRemove(item) {
    actions.remove(item);
    closeOnNarrow();
  }

  return (
    <aside
      className={`chat-sidebar${railCollapsed ? " is-collapsed" : ""}${
        mobileOpen ? " is-drawer-open" : ""
      }`}
      id="chat-sidebar"
      ref={sidebarRef}
      aria-label="محادثات المساعد"
      aria-hidden={isMobile && !mobileOpen ? "true" : undefined}
      inert={isMobile && !mobileOpen ? true : undefined}
      tabIndex={-1}
    >
      <div className="chat-sidebar-inner" id="chat-sidebar-content">
        <div className="chat-sidebar-head">
          <button
            type="button"
            className="chat-sidebar-toggle"
            onClick={handleToggle}
            aria-label={toggleLabel}
            title={toggleLabel}
            aria-expanded={isMobile ? Boolean(mobileOpen) : !railCollapsed}
            aria-controls="chat-sidebar-content"
          >
            <i className="bi bi-layout-sidebar-inset" aria-hidden="true"></i>
          </button>

          <div
            className="chat-sidebar-heading"
            aria-hidden={railCollapsed ? "true" : undefined}
            inert={railCollapsed ? true : undefined}
          >
            <p className="chat-sidebar-eyebrow">المساعد المعرفي</p>
            <h2>المحادثات</h2>
          </div>
        </div>

        <button
          type="button"
          className="chat-sidebar-new"
          onClick={handleNewChat}
          disabled={busy}
          title="محادثة جديدة"
        >
          <i className="bi bi-plus-lg" aria-hidden="true"></i>
          <span aria-hidden={railCollapsed ? "true" : undefined}>
            محادثة جديدة
          </span>
        </button>

        {user ? (
          <div className="chat-sidebar-search">
            <button
              type="button"
              className="chat-sidebar-search-trigger"
              onClick={handleSearchActivate}
              aria-label="البحث في المحادثات"
              title="البحث في المحادثات"
            >
              <i className="bi bi-search" aria-hidden="true"></i>
            </button>

            <span className="sr-only">البحث في المحادثات</span>

            <input
              ref={searchRef}
              type="search"
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
              placeholder="ابحث في المحادثات…"
              aria-label="البحث في المحادثات"
              aria-hidden={railCollapsed ? "true" : undefined}
              inert={railCollapsed ? true : undefined}
              tabIndex={railCollapsed ? -1 : undefined}
              autoComplete="off"
            />

            {searchQuery ? (
              <button
                type="button"
                className="chat-sidebar-search-clear"
                onClick={() => setSearchQuery("")}
                aria-label="مسح البحث"
                title="مسح البحث"
                aria-hidden={railCollapsed ? "true" : undefined}
                inert={railCollapsed ? true : undefined}
                tabIndex={railCollapsed ? -1 : undefined}
              >
                <i className="bi bi-x-circle" aria-hidden="true"></i>
              </button>
            ) : null}
          </div>
        ) : null}

        <button
          type="button"
          className="chat-sidebar-section-label"
          onClick={handleHistoryActivate}
          aria-label="السجل"
          title="السجل"
        >
          <span className="chat-sidebar-section-icon" aria-hidden="true">
            <i className="bi bi-clock-history"></i>
          </span>
          <span className="chat-sidebar-section-text">السجل</span>
          <span className="chat-sidebar-section-count">
            {formatNumber(visibleConversations.length)}
          </span>
        </button>

        <div
          className="chat-sidebar-body"
          ref={bodyRef}
          aria-hidden={railCollapsed ? "true" : undefined}
          inert={railCollapsed ? true : undefined}
        >
          {!user ? (
            <div className="chat-sidebar-guest">
              <i className="bi bi-chat-square-text" aria-hidden="true"></i>
              <p>
                سجّل الدخول لحفظ محادثاتك والرجوع إليها في أي وقت. يمكنك
                الاستمرار كزائر الآن، لكن المحادثة لن تُحفظ بعد إغلاق الصفحة.
              </p>
              <Link className="btn btn-secondary btn-sm" to="/signin">
                تسجيل الدخول
              </Link>
            </div>
          ) : !loaded ? (
            <div className="chat-sidebar-skeleton" aria-hidden="true">
              <span className="skeleton"></span>
              <span className="skeleton"></span>
              <span className="skeleton"></span>
            </div>
          ) : visibleConversations.length === 0 ? (
            <p className="chat-sidebar-empty">
              {query
                ? "لا توجد محادثات بهذا الاسم."
                : "لا توجد محادثات محفوظة بعد. ابدأ محادثة جديدة."}
            </p>
          ) : (
            <div
              className="chat-sidebar-history"
              aria-label="المحادثات الحديثة"
            >
              {visibleConversations.map((item) => {
                const isActive = String(activeId) === String(item._id);
                const title = conversationTitle(item);

                return (
                  <div
                    className={`chat-sidebar-row${
                      isActive ? " is-active" : ""
                    }`}
                    key={item._id}
                  >
                    <button
                      type="button"
                      className={`chat-sidebar-item${
                        isActive ? " is-active" : ""
                      }`}
                      onClick={() => handleOpenConversation(item)}
                      disabled={busy}
                      aria-current={isActive ? "true" : undefined}
                      title={title}
                    >
                      <i
                        className="bi bi-chat-left-text"
                        aria-hidden="true"
                      ></i>
                      <span>{title}</span>
                    </button>

                    <div className="chat-sidebar-actions">
                      <button
                        type="button"
                        className="chat-sidebar-action"
                        onClick={() => handleRename(item)}
                        disabled={busy}
                        aria-label={`تغيير اسم المحادثة: ${title}`}
                        title="تغيير الاسم"
                      >
                        <i
                          className="bi bi-pencil-square"
                          aria-hidden="true"
                        ></i>
                      </button>
                      <button
                        type="button"
                        className="chat-sidebar-action danger"
                        onClick={() => handleRemove(item)}
                        disabled={busy}
                        aria-label={`حذف المحادثة: ${title}`}
                        title="حذف"
                      >
                        <i className="bi bi-trash3" aria-hidden="true"></i>
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        <div className="chat-sidebar-foot">
          <div className="chat-sidebar-theme">
            <span
              className="chat-sidebar-theme-label"
              aria-hidden={railCollapsed ? "true" : undefined}
            >
              <i className="bi bi-circle-half" aria-hidden="true"></i>
              المظهر
            </span>
            <ThemeToggle />
          </div>

          {authLoading ? (
            <span className="chat-sidebar-user-loading" aria-hidden="true" />
          ) : user ? (
            <Link
              className="chat-sidebar-user"
              to="/profile"
              title={displayName}
              aria-label={`الحساب: ${displayName}`}
              onClick={closeOnNarrow}
            >
              <span className="avatar" aria-hidden="true">
                {initials(displayName, "ب")}
              </span>
              <span
                className="chat-sidebar-user-copy"
                aria-hidden={railCollapsed ? "true" : undefined}
              >
                <strong>{displayName}</strong>
                <span dir="ltr">{user.email}</span>
              </span>
              <i
                className="bi bi-sliders chat-sidebar-user-icon"
                aria-hidden="true"
              ></i>
            </Link>
          ) : (
            <button
              type="button"
              className="chat-sidebar-signin"
              onClick={() => {
                navigate("/signin");
                closeOnNarrow();
              }}
              aria-label="تسجيل الدخول"
              title="تسجيل الدخول"
            >
              <i className="bi bi-person" aria-hidden="true"></i>
              <span aria-hidden={railCollapsed ? "true" : undefined}>
                تسجيل الدخول
              </span>
            </button>
          )}
        </div>
      </div>
    </aside>
  );
}

export default ChatSidebar;
