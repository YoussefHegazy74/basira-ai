import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import api from "../services/api";
import { useAuth } from "../context/AuthContext";
import {
  buildChatHistory,
  makeClientMessageId,
  mapStoredMessages,
  mergeServerMessages,
  readActiveConversation,
  readFailedSubmission,
  writeActiveConversation,
  writeFailedSubmission,
} from "../utils/chatHistory";
import { citationSegments } from "../utils/citations";
import { arabicMessage, formatNumber } from "../utils/format";
import Modal, { ConfirmDialog } from "../components/Modal";
import "../styles/chat.css";

function conversationTitle(item) {
  return item?.title || "محادثة جديدة";
}

function conversationCountLabel(count) {
  if (count === 0) return "لا محادثات محفوظة";
  if (count === 1) return "محادثة واحدة";
  if (count === 2) return "محادثتان";
  if (count <= 10) return `${formatNumber(count)} محادثات`;
  return `${formatNumber(count)} محادثة`;
}

function fallbackCopy(text) {
  const field = document.createElement("textarea");
  field.value = text;
  field.setAttribute("readonly", "");
  field.style.position = "fixed";
  field.style.insetBlockStart = "-9999px";
  document.body.appendChild(field);
  field.select();

  try {
    document.execCommand("copy");
  } finally {
    document.body.removeChild(field);
  }
}

function Chat() {
  const { user, authLoading } = useAuth();

  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [conversationId, setConversationId] = useState(null);
  const [conversations, setConversations] = useState([]);
  const [historyLoaded, setHistoryLoaded] = useState(false);
  const [isOpeningConversation, setIsOpeningConversation] = useState(false);
  const [editingConversationId, setEditingConversationId] = useState(null);
  const [deletingConversationId, setDeletingConversationId] = useState(null);
  const [failedSubmission, setFailedSubmission] = useState(null);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [copiedId, setCopiedId] = useState("");
  const [showScrollDown, setShowScrollDown] = useState(false);

  const [renameTarget, setRenameTarget] = useState(null);
  const [renameValue, setRenameValue] = useState("");
  const [renameError, setRenameError] = useState("");
  const [deleteTarget, setDeleteTarget] = useState(null);

  const textareaRef = useRef(null);
  const messagesRef = useRef(null);
  const stickToBottomRef = useRef(true);
  const jumpToBottomRef = useRef(false);
  const restoredConversationRef = useRef(false);
  const copyTimerRef = useRef(null);

  const userIdentity = user?.id || user?._id || user?.email || null;
  const maxLength = user ? 500 : 250;

  // الحفاظ على موضع الكتابة أثناء نمو الحقل.
  useEffect(
    function () {
      const field = textareaRef.current;
      if (!field) return;

      field.style.height = "auto";
      const nextHeight = Math.min(field.scrollHeight, 168);
      field.style.height = `${nextHeight}px`;
      field.style.overflowY =
        field.scrollHeight > 168 ? "auto" : "hidden";
    },
    [message]
  );

  // تمرير هادئ في نهاية الرسائل، مع احترام تمرير المستخدم للأعلى.
  useEffect(
    function () {
      const container = messagesRef.current;
      if (!container || !stickToBottomRef.current) return;

      const reduced =
        typeof window !== "undefined" &&
        window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches;

      container.scrollTo({
        top: container.scrollHeight,
        behavior:
          jumpToBottomRef.current || reduced ? "auto" : "smooth",
      });
      jumpToBottomRef.current = false;
    },
    [messages, isLoading]
  );

  useEffect(
    function () {
      return function () {
        if (copyTimerRef.current) {
          window.clearTimeout(copyTimerRef.current);
        }
      };
    },
    []
  );

  // تحميل سجل المحادثات للمستخدم المسجل فقط
  const loadConversations = useCallback(
    async function () {
      if (!user) {
        setConversations([]);
        setHistoryLoaded(true);
        return;
      }

      try {
        const response = await api.get("/conversations?page=1&limit=50");
        const data = response.data?.data;

        const list = Array.isArray(data)
          ? data
          : Array.isArray(data?.conversations)
            ? data.conversations
            : [];

        setConversations(list);
      } catch (requestError) {
        console.error(
          "Failed to load conversations:",
          requestError.response?.data || requestError.message
        );
      } finally {
        setHistoryLoaded(true);
      }
    },
    [user]
  );

  useEffect(
    function () {
      if (authLoading) return;

      if (user) {
        loadConversations();
      } else {
        restoredConversationRef.current = false;
        setConversations([]);
        setHistoryLoaded(true);
        setConversationId(null);
        setMessages([]);
        setFailedSubmission(null);
        setHistoryOpen(false);
      }
    },
    [authLoading, user, loadConversations]
  );

  useEffect(
    function () {
      restoredConversationRef.current = false;
      setHistoryLoaded(false);
    },
    [userIdentity]
  );

  // استعادة المحادثة النشطة بعد إعادة تحميل الصفحة.
  useEffect(
    function () {
      if (authLoading || !user) return;
      if (restoredConversationRef.current || conversationId) return;

      restoredConversationRef.current = true;
      const savedId = readActiveConversation(user);
      if (!savedId) return;

      let cancelled = false;
      setIsOpeningConversation(true);

      (async function () {
        try {
          const response = await api.get(`/conversations/${savedId}`);
          const data = response.data?.data;
          if (!data?.conversation) {
            throw new Error("تعذر استعادة المحادثة السابقة.");
          }
          if (cancelled) return;

          setIsOpeningConversation(false);
          const storedMessages = mapStoredMessages(data.messages);
          jumpToBottomRef.current = true;
          stickToBottomRef.current = true;
          setConversationId(data.conversation._id);
          setMessages(storedMessages);
          setMessage("");

          const savedFailure = readFailedSubmission(user);
          if (
            savedFailure &&
            String(savedFailure.conversationId) ===
              String(data.conversation._id)
          ) {
            setFailedSubmission({
              ...savedFailure,
              historyMessages: storedMessages,
            });
          }
        } catch (requestError) {
          if (cancelled) return;
          setIsOpeningConversation(false);
          writeActiveConversation(user, null);
          setConversationId(null);
          setMessages([]);
          if (requestError.response?.status !== 404) {
            setError(
              arabicMessage(
                requestError.response?.data?.message,
                "تعذّر استعادة المحادثة السابقة. يمكنك بدء محادثة جديدة."
              )
            );
          }
        }
      })();

      return function () {
        cancelled = true;
      };
      // نعتمد على هوية ثابتة بدل كائن المستخدم: يتبدل الكائن بعد نداء
      // الاستخدام، وهذا قد يلغي الاستعادة ويترك حالة التحميل معلقة.
    },
    [authLoading, userIdentity, conversationId]
  );

  // فتح محادثة محفوظة
  async function openConversation(item) {
    if (
      !user ||
      isLoading ||
      isOpeningConversation ||
      editingConversationId ||
      deletingConversationId
    ) {
      return;
    }

    setError("");
    setHistoryOpen(false);
    setIsOpeningConversation(true);

    try {
      const response = await api.get(`/conversations/${item._id}`);
      const data = response.data?.data;

      if (!data?.conversation) {
        throw new Error("تعذر تحميل المحادثة.");
      }

      const storedMessages = mapStoredMessages(data.messages);
      jumpToBottomRef.current = true;
      stickToBottomRef.current = true;
      setConversationId(data.conversation._id);
      setMessages(storedMessages);
      setMessage("");
      writeActiveConversation(user, data.conversation._id);

      const savedFailure = readFailedSubmission(user);
      if (
        savedFailure &&
        String(savedFailure.conversationId) ===
          String(data.conversation._id)
      ) {
        setFailedSubmission({
          ...savedFailure,
          historyMessages: storedMessages,
        });
      } else {
        setFailedSubmission(null);
      }
    } catch (requestError) {
      setError(
        arabicMessage(
          requestError.response?.data?.message,
          "تعذّر فتح المحادثة. حاول مرة أخرى."
        )
      );
    } finally {
      setIsOpeningConversation(false);
    }
  }

  function openRenameDialog(item) {
    if (
      !user ||
      isLoading ||
      isOpeningConversation ||
      editingConversationId ||
      deletingConversationId
    ) {
      return;
    }

    setRenameTarget(item);
    setRenameValue(conversationTitle(item));
    setRenameError("");
  }

  function closeRenameDialog() {
    if (editingConversationId) return;
    setRenameTarget(null);
    setRenameError("");
  }

  // تغيير اسم المحادثة
  async function handleRenameSubmit(event) {
    event?.preventDefault();

    if (!renameTarget || editingConversationId) return;

    const currentTitle = conversationTitle(renameTarget);
    const trimmedTitle = renameValue.trim();

    if (!trimmedTitle) {
      setRenameError("اسم المحادثة لا يمكن أن يكون فارغًا.");
      return;
    }

    if (trimmedTitle.length > 100) {
      setRenameError("اجعل الاسم أقصر من مئة حرف.");
      return;
    }

    if (trimmedTitle === currentTitle) {
      setRenameTarget(null);
      return;
    }

    setRenameError("");
    setEditingConversationId(renameTarget._id);

    try {
      await api.put(`/conversations/${renameTarget._id}`, {
        newTitle: trimmedTitle,
      });

      setConversations(function (currentConversations) {
        return currentConversations.map(function (conversation) {
          if (String(conversation._id) === String(renameTarget._id)) {
            return {
              ...conversation,
              title: trimmedTitle,
            };
          }

          return conversation;
        });
      });

      await loadConversations();
      setRenameTarget(null);
    } catch (requestError) {
      console.error(
        "Failed to rename conversation:",
        requestError.response?.data || requestError.message
      );

      setRenameError(
        arabicMessage(
          requestError.response?.data?.message,
          "تعذّر تغيير اسم المحادثة. حاول مرة أخرى."
        )
      );
    } finally {
      setEditingConversationId(null);
    }
  }

  function openDeleteDialog(item) {
    if (
      !user ||
      isLoading ||
      isOpeningConversation ||
      editingConversationId ||
      deletingConversationId
    ) {
      return;
    }

    setDeleteTarget(item);
  }

  // حذف المحادثة ورسائلها من السجل
  async function handleDeleteConversation() {
    if (!deleteTarget || deletingConversationId) return;

    const item = deleteTarget;
    setError("");
    setDeletingConversationId(item._id);

    try {
      await api.delete(`/conversations/${item._id}`);

      setConversations(function (currentConversations) {
        return currentConversations.filter(function (conversation) {
          return String(conversation._id) !== String(item._id);
        });
      });

      if (String(conversationId) === String(item._id)) {
        setConversationId(null);
        setMessages([]);
        setMessage("");
        setFailedSubmission(null);
        writeActiveConversation(user, null);
        writeFailedSubmission(user, null);
      }

      await loadConversations();
      setDeleteTarget(null);
    } catch (requestError) {
      console.error(
        "Failed to delete conversation:",
        requestError.response?.data || requestError.message
      );

      setError(
        arabicMessage(
          requestError.response?.data?.message,
          "تعذّر حذف المحادثة. حاول مرة أخرى."
        )
      );
      setDeleteTarget(null);
    } finally {
      setDeletingConversationId(null);
    }
  }

  function errorMessage(requestError) {
    return arabicMessage(
      requestError.response?.data?.message ||
        requestError.response?.data?.error?.message,
      "تعذّر الاتصال بالخدمة الآن. تحقق من اتصالك ثم حاول مرة أخرى."
    );
  }

  async function requestChat({
    text,
    clientMessageId,
    targetConversationId,
    historyMessages,
  }) {
    const requestBody = {
      content: text,
      clientMessageId,
      chat_history: buildChatHistory(historyMessages, clientMessageId),
    };

    // المحادثات المحفوظة متاحة للمستخدم المسجل فقط.
    if (user && targetConversationId) {
      requestBody.conversationId = targetConversationId;
    }

    const response = await api.post("/messages", requestBody);
    return response.data?.data || {};
  }

  function mergeServerResponse(
    data,
    clientMessageId,
    { appendUser = false } = {}
  ) {
    if (user && data.conversation?._id) {
      setConversationId(data.conversation._id);
      writeActiveConversation(user, data.conversation._id);
    }

    setMessages(function (currentMessages) {
      return mergeServerMessages(currentMessages, data, clientMessageId, {
        appendUser,
      });
    });
    writeFailedSubmission(user, null);
  }

  function handleSendFailure(
    requestError,
    { text, clientMessageId, targetConversationId, historyMessages }
  ) {
    console.error(
      "Failed to send message:",
      requestError.response?.data || requestError.message
    );

    const recovery = requestError.response?.data?.data || null;

    if (recovery?.conversationId) {
      setConversationId(recovery.conversationId);
      writeActiveConversation(user, recovery.conversationId);
    }

    if (recovery?.userMessageId) {
      setMessages(function (currentMessages) {
        return currentMessages.map(function (item) {
          if (item.clientMessageId !== clientMessageId) return item;
          return { ...item, id: recovery.userMessageId };
        });
      });
    }

    const retryable = Boolean(recovery?.retryable) || !requestError.response;

    const failed = retryable
      ? {
          text,
          clientMessageId,
          historyMessages,
          conversationId:
            recovery?.conversationId || targetConversationId || null,
        }
      : null;
    setFailedSubmission(failed);
    writeFailedSubmission(
      user,
      failed
        ? {
            text: failed.text,
            clientMessageId: failed.clientMessageId,
            conversationId: failed.conversationId,
          }
        : null
    );
    setError(errorMessage(requestError));
  }

  async function handleSubmit(event) {
    event.preventDefault();

    const text = message.trim();

    if (!text || isLoading || isOpeningConversation) return;

    if (text.length > maxLength) {
      setError(
        `الحد الأقصى لطول الرسالة هو ${formatNumber(maxLength)} حرفًا.`
      );
      return;
    }

    setError("");
    setIsLoading(true);
    setMessage("");
    setFailedSubmission(null);
    stickToBottomRef.current = true;

    const clientMessageId = makeClientMessageId();
    const historyMessages = messages;
    const targetConversationId = conversationId;

    setMessages(function (currentMessages) {
      return [
        ...currentMessages,
        {
          id: `local-user-${clientMessageId}`,
          clientMessageId,
          text,
          sender: "user",
        },
      ];
    });

    try {
      const data = await requestChat({
        text,
        clientMessageId,
        targetConversationId,
        historyMessages,
      });

      mergeServerResponse(data, clientMessageId);

      if (user) {
        await loadConversations();
      }
    } catch (requestError) {
      handleSendFailure(requestError, {
        text,
        clientMessageId,
        targetConversationId,
        historyMessages,
      });
    } finally {
      setIsLoading(false);
      textareaRef.current?.focus();
    }
  }

  async function retryFailedSubmission() {
    if (!failedSubmission || isLoading || isOpeningConversation) return;

    const failed = failedSubmission;
    setError("");
    setIsLoading(true);
    stickToBottomRef.current = true;

    try {
      const data = await requestChat({
        text: failed.text,
        clientMessageId: failed.clientMessageId,
        targetConversationId: failed.conversationId,
        historyMessages: failed.historyMessages,
      });

      mergeServerResponse(data, failed.clientMessageId, {
        appendUser: true,
      });
      setFailedSubmission(null);

      if (user) {
        await loadConversations();
      }
    } catch (requestError) {
      handleSendFailure(requestError, {
        text: failed.text,
        clientMessageId: failed.clientMessageId,
        targetConversationId: failed.conversationId,
        historyMessages: failed.historyMessages,
      });
    } finally {
      setIsLoading(false);
      textareaRef.current?.focus();
    }
  }

  function handleKeyDown(event) {
    if (event.nativeEvent?.isComposing) return;

    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  }

  function handleMessagesScroll() {
    const container = messagesRef.current;
    if (!container) return;

    const distanceFromBottom =
      container.scrollHeight - container.scrollTop - container.clientHeight;
    const nearBottom = distanceFromBottom < 96;

    stickToBottomRef.current = nearBottom;
    setShowScrollDown(
      !nearBottom && container.scrollHeight > container.clientHeight + 160
    );
  }

  function scrollToLatest() {
    const container = messagesRef.current;
    if (!container) return;

    stickToBottomRef.current = true;
    setShowScrollDown(false);

    const reduced = window.matchMedia?.(
      "(prefers-reduced-motion: reduce)"
    )?.matches;

    container.scrollTo({
      top: container.scrollHeight,
      behavior: reduced ? "auto" : "smooth",
    });
  }

  async function handleCopy(item) {
    const text = item?.text || "";
    if (!text) return;

    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(text);
      } else {
        fallbackCopy(text);
      }

      setCopiedId(item.id);
      if (copyTimerRef.current) {
        window.clearTimeout(copyTimerRef.current);
      }
      copyTimerRef.current = window.setTimeout(function () {
        setCopiedId("");
      }, 2000);
    } catch {
      setError("تعذّر نسخ الرسالة من المتصفح. حدّد النص وانسخه يدويًا.");
    }
  }

  // بدء محادثة جديدة
  function handleNewChat() {
    if (
      isLoading ||
      isOpeningConversation ||
      editingConversationId ||
      deletingConversationId
    ) {
      return;
    }

    setMessages([]);
    setMessage("");
    setError("");
    setConversationId(null);
    setFailedSubmission(null);
    setHistoryOpen(false);
    setShowScrollDown(false);
    stickToBottomRef.current = true;
    writeActiveConversation(user, null);
    writeFailedSubmission(user, null);

    textareaRef.current?.focus();
  }

  function chooseSuggestion(suggestion) {
    if (isLoading || isOpeningConversation) return;
    setMessage(suggestion);
    setError("");
    textareaRef.current?.focus();
  }

  const statusText = isOpeningConversation
    ? "جارٍ تحميل المحادثة…"
    : isLoading
      ? "بصيرة تكتب الآن…"
      : "جاهز للمحادثة";

  const busyHistory =
    isLoading ||
    isOpeningConversation ||
    Boolean(editingConversationId) ||
    Boolean(deletingConversationId);

  const deleteBusy =
    Boolean(deleteTarget) &&
    String(deletingConversationId) === String(deleteTarget?._id);

  return (
    <main className="page chat-page">
      <div className="container chat-shell">
        <header className="chat-intro">
          <div>
            <p className="eyebrow">المحادثة</p>
            <h1>المساعد المعرفي</h1>
            <p>
              اسأل في مسألة شرعية، وتابع الإجابة بهدوء. يجيب بصيرة من
              مصادره المتاحة، ويوضح لك إذا لم يجد الجواب فيها.
            </p>
          </div>

          {!authLoading && !user ? (
            <p className="chat-guest-note">
              <i className="bi bi-info-circle" aria-hidden="true"></i>
              <span>
                أنت تستخدم بصيرة كزائر.{" "}
                <Link to="/signin">سجّل الدخول</Link> لحفظ محادثاتك.
              </span>
            </p>
          ) : null}
        </header>

        <div className={`chat-layout ${user ? "" : "is-guest"}`}>
          {user ? (
            <aside
              className={`chat-sidebar ${historyOpen ? "is-open" : ""}`}
              id="chat-history"
              aria-label="سجل المحادثات"
            >
              <div className="chat-sidebar-heading">
                <div>
                  <h2>سجل المحادثات</h2>
                  <p>{conversationCountLabel(conversations.length)}</p>
                </div>
                <button
                  className="icon-btn chat-history-close"
                  type="button"
                  onClick={() => setHistoryOpen(false)}
                  aria-label="إغلاق سجل المحادثات"
                >
                  <i className="bi bi-x-lg" aria-hidden="true"></i>
                </button>
              </div>

              <button
                className="btn btn-secondary chat-new-button"
                type="button"
                onClick={handleNewChat}
                disabled={busyHistory}
              >
                <i className="bi bi-plus-lg" aria-hidden="true"></i>
                محادثة جديدة
              </button>

              <div className="chat-sidebar-list">
                {!historyLoaded ? (
                  <div
                    className="chat-history-skeleton"
                    aria-hidden="true"
                  >
                    <span className="skeleton"></span>
                    <span className="skeleton"></span>
                    <span className="skeleton"></span>
                  </div>
                ) : conversations.length === 0 ? (
                  <div className="chat-sidebar-empty">
                    <i
                      className="bi bi-chat-square-text"
                      aria-hidden="true"
                    ></i>
                    <p>لا توجد محادثات محفوظة بعد.</p>
                    <span>ستظهر هنا بعد أول محادثة.</span>
                  </div>
                ) : (
                  conversations.map(function (item) {
                    const isActive =
                      String(conversationId) === String(item._id);
                    const isRenaming =
                      String(editingConversationId) === String(item._id);
                    const isDeleting =
                      String(deletingConversationId) === String(item._id);

                    return (
                      <div
                        className={`chat-history-row ${
                          isActive ? "active" : ""
                        }`}
                        key={item._id}
                      >
                        <button
                          type="button"
                          className={`chat-history-item ${
                            isActive ? "active" : ""
                          }`}
                          onClick={function () {
                            openConversation(item);
                          }}
                          disabled={busyHistory}
                          aria-current={isActive ? "true" : undefined}
                          title={conversationTitle(item)}
                        >
                          <i
                            className="bi bi-chat-left-text"
                            aria-hidden="true"
                          ></i>
                          <span>{conversationTitle(item)}</span>
                        </button>

                        <div className="chat-history-actions">
                          <button
                            type="button"
                            className="chat-history-action"
                            title="تغيير اسم المحادثة"
                            aria-label={`تغيير اسم المحادثة: ${conversationTitle(
                              item
                            )}`}
                            onClick={function () {
                              openRenameDialog(item);
                            }}
                            disabled={busyHistory}
                          >
                            <i
                              className={
                                isRenaming
                                  ? "bi bi-hourglass-split"
                                  : "bi bi-pencil-square"
                              }
                              aria-hidden="true"
                            ></i>
                          </button>

                          <button
                            type="button"
                            className="chat-history-action danger"
                            title="حذف المحادثة"
                            aria-label={`حذف المحادثة: ${conversationTitle(
                              item
                            )}`}
                            onClick={function () {
                              openDeleteDialog(item);
                            }}
                            disabled={busyHistory}
                          >
                            <i
                              className={
                                isDeleting
                                  ? "bi bi-hourglass-split"
                                  : "bi bi-trash3"
                              }
                              aria-hidden="true"
                            ></i>
                          </button>
                        </div>
                      </div>
                    );
                  })
                )}
              </div>
            </aside>
          ) : null}

          <section className="chat-workspace" aria-label="المحادثة">
            <header className="chat-header">
              <div className="chat-header-info">
                <span className="chat-avatar" aria-hidden="true">
                  <i className="bi bi-eye"></i>
                </span>

                <div className="chat-header-copy">
                  <h2>بصيرة</h2>
                  <span className="chat-status">
                    <span
                      className={`status-dot ${
                        isLoading || isOpeningConversation ? "busy" : ""
                      }`}
                    ></span>
                    {statusText}
                  </span>
                </div>
              </div>

              <div className="chat-header-actions">
                {user ? (
                  <button
                    type="button"
                    className="btn btn-ghost btn-sm chat-history-toggle"
                    onClick={() => setHistoryOpen((current) => !current)}
                    aria-expanded={historyOpen}
                    aria-controls="chat-history"
                    aria-label="عرض سجل المحادثات"
                  >
                    <i className="bi bi-clock-history" aria-hidden="true"></i>
                    <span className="btn-label">السجل</span>
                  </button>
                ) : null}

                <button
                  type="button"
                  className="btn btn-secondary btn-sm"
                  onClick={handleNewChat}
                  disabled={busyHistory}
                  aria-label="بدء محادثة جديدة"
                >
                  <i className="bi bi-plus-lg" aria-hidden="true"></i>
                  <span className="btn-label">محادثة جديدة</span>
                </button>
              </div>
            </header>

            <div
              className="chat-messages"
              ref={messagesRef}
              onScroll={handleMessagesScroll}
              aria-label="الرسائل"
            >
              {messages.length === 0 ? (
                isOpeningConversation ? (
                  <div className="chat-loading-state" aria-live="polite">
                    <span className="spinner" aria-hidden="true"></span>
                    <p>جارٍ تحميل المحادثة…</p>
                  </div>
                ) : (
                  <div className="chat-empty-state">
                    <span className="chat-empty-icon" aria-hidden="true">
                      <i className="bi bi-chat-quote"></i>
                    </span>
                    <h3>اسأل في مسألة شرعية</h3>
                    <p>
                      اكتب سؤالك الشرعي في الأسفل. يجيب بصيرة من المصادر
                      المتاحة فقط، ويمكنك البدء بأحد الأسئلة المقترحة.
                    </p>

                    <div className="chat-suggestions">
                      <button
                        type="button"
                        onClick={() =>
                          chooseSuggestion("ما حكم صلاة المسافر؟")
                        }
                      >
                        ما حكم صلاة المسافر؟
                      </button>
                      <button
                        type="button"
                        onClick={() =>
                          chooseSuggestion(
                            "ما تفسير قوله تعالى: إن مع العسر يسرا؟"
                          )
                        }
                      >
                        ما تفسير آية «إن مع العسر يسرا»؟
                      </button>
                      <button
                        type="button"
                        onClick={() =>
                          chooseSuggestion(
                            "ما الفرق بين الحديث الصحيح والضعيف؟"
                          )
                        }
                      >
                        ما الفرق بين الحديث الصحيح والضعيف؟
                      </button>
                    </div>
                  </div>
                )
              ) : (
                <div className="message-list">
                  {messages.map(function (item) {
                    const isCopied = copiedId === item.id;

                    return (
                      <div
                        className={`message-row ${item.sender}`}
                        key={item.id}
                      >
                        {item.sender === "assistant" ? (
                          <span
                            className="message-avatar"
                            aria-hidden="true"
                          >
                            <i className="bi bi-eye"></i>
                          </span>
                        ) : null}

                        <div className="message-stack">
                          <div className="message-meta">
                            <span className="message-sender">
                              {item.sender === "user" ? "أنت" : "بصيرة"}
                            </span>
                            <button
                              type="button"
                              className={`message-copy ${
                                isCopied ? "is-copied" : ""
                              }`}
                              onClick={() => handleCopy(item)}
                              title={
                                isCopied ? "تم نسخ الرسالة" : "نسخ الرسالة"
                              }
                              aria-label={
                                isCopied ? "تم نسخ الرسالة" : "نسخ الرسالة"
                              }
                            >
                              <i
                                className={
                                  isCopied
                                    ? "bi bi-check2"
                                    : "bi bi-copy"
                                }
                                aria-hidden="true"
                              ></i>
                            </button>
                          </div>

                          <div className="message-bubble">
                            <p>
                              {citationSegments(
                                item.text,
                                item.sources
                              ).map(function (segment, index) {
                                if (segment.type === "citation") {
                                  return (
                                    <span
                                      className="inline-citation"
                                      key={`${item.id}-citation-${index}`}
                                    >
                                      {segment.text}
                                    </span>
                                  );
                                }
                                return (
                                  <span
                                    key={`${item.id}-text-${index}`}
                                  >
                                    {segment.text}
                                  </span>
                                );
                              })}
                            </p>
                          </div>
                        </div>
                      </div>
                    );
                  })}

                  {isLoading ? (
                    <div className="message-row assistant typing-row">
                      <span className="message-avatar" aria-hidden="true">
                        <i className="bi bi-eye"></i>
                      </span>

                      <div className="message-stack">
                        <span className="message-sender">بصيرة</span>
                        <div
                          className="message-bubble typing-bubble"
                          role="status"
                        >
                          <span
                            className="typing-dots"
                            aria-hidden="true"
                          >
                            <span></span>
                            <span></span>
                            <span></span>
                          </span>
                          <span className="sr-only">
                            بصيرة تكتب الآن…
                          </span>
                        </div>
                      </div>
                    </div>
                  ) : null}
                </div>
              )}
            </div>

            {showScrollDown ? (
              <button
                type="button"
                className="chat-scroll-down"
                onClick={scrollToLatest}
                aria-label="الانتقال إلى آخر الرسائل"
                title="آخر الرسائل"
              >
                <i className="bi bi-arrow-down" aria-hidden="true"></i>
              </button>
            ) : null}

            {error || failedSubmission ? (
              <div className="chat-alert" role="alert">
                <i
                  className="bi bi-exclamation-circle"
                  aria-hidden="true"
                ></i>
                <div className="chat-alert-copy">
                  <strong>
                    {failedSubmission
                      ? "لم يكتمل الرد السابق"
                      : "تعذّر إكمال الطلب"}
                  </strong>
                  <p>
                    {error ||
                      "يمكنك إعادة المحاولة، أو كتابة سؤالك من جديد."}
                  </p>
                </div>
                {failedSubmission ? (
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    onClick={retryFailedSubmission}
                    disabled={isLoading || isOpeningConversation}
                  >
                    {isLoading ? (
                      <>
                        <span
                          className="spinner"
                          aria-hidden="true"
                        ></span>
                        جارٍ الإرسال…
                      </>
                    ) : (
                      <>
                        <i
                          className="bi bi-arrow-clockwise"
                          aria-hidden="true"
                        ></i>
                        إعادة المحاولة
                      </>
                    )}
                  </button>
                ) : null}
              </div>
            ) : null}

            <form className="chat-composer" onSubmit={handleSubmit}>
              <div className="composer-box">
                <label className="sr-only" htmlFor="chat-message">
                  رسالتك
                </label>

                <textarea
                  id="chat-message"
                  ref={textareaRef}
                  value={message}
                  onChange={function (event) {
                    setMessage(event.target.value);
                    if (error) setError("");
                    if (failedSubmission) {
                      setFailedSubmission(null);
                      writeFailedSubmission(user, null);
                    }
                  }}
                  onKeyDown={handleKeyDown}
                  placeholder="اكتب سؤالك هنا…"
                  rows={1}
                  maxLength={maxLength}
                  disabled={isOpeningConversation}
                  aria-describedby="chat-composer-hint"
                />

                <div className="composer-footer">
                  <span className="composer-meta">
                    {message.length > 0 ? (
                      <span
                        className={
                          message.length >= maxLength
                            ? "composer-count limit"
                            : "composer-count"
                        }
                      >
                        {formatNumber(message.length)} /{" "}
                        {formatNumber(maxLength)}
                      </span>
                    ) : null}
                    <span className="composer-hint" id="chat-composer-hint">
                      مفتاح الإدخال يرسل، ومعه مفتاح التبديل يضيف سطرًا.
                    </span>
                  </span>

                  <button
                    className="btn btn-primary send-message-button"
                    type="submit"
                    disabled={
                      !message.trim() ||
                      isLoading ||
                      isOpeningConversation
                    }
                    aria-busy={isLoading}
                    aria-label={isLoading ? "جارٍ الإرسال" : "إرسال الرسالة"}
                  >
                    {isLoading ? (
                      <>
                        <span className="spinner" aria-hidden="true"></span>
                        <span className="send-label">جارٍ الإرسال…</span>
                      </>
                    ) : (
                      <>
                        <span className="send-label">إرسال</span>
                        <i
                          className="bi bi-send"
                          aria-hidden="true"
                        ></i>
                      </>
                    )}
                  </button>
                </div>
              </div>
            </form>
          </section>
        </div>
      </div>

      {renameTarget ? (
        <Modal
          title="تغيير اسم المحادثة"
          description="اختر اسمًا قصيرًا يسهّل العثور على المحادثة في السجل."
          icon="bi-pencil-square"
          onClose={closeRenameDialog}
          busy={Boolean(editingConversationId)}
          actions={
            <>
              <button
                type="submit"
                form="rename-conversation-form"
                className="btn btn-primary"
                disabled={Boolean(editingConversationId)}
              >
                {editingConversationId ? (
                  <>
                    <span className="spinner" aria-hidden="true"></span>
                    جارٍ الحفظ…
                  </>
                ) : (
                  "حفظ الاسم"
                )}
              </button>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={closeRenameDialog}
                disabled={Boolean(editingConversationId)}
              >
                إلغاء
              </button>
            </>
          }
        >
          <form
            id="rename-conversation-form"
            className="modal-form"
            onSubmit={handleRenameSubmit}
            noValidate
          >
            <div className="field">
              <label htmlFor="conversation-title">اسم المحادثة</label>
              <input
                id="conversation-title"
                type="text"
                value={renameValue}
                onChange={function (event) {
                  setRenameValue(event.target.value);
                  if (renameError) setRenameError("");
                }}
                maxLength={100}
                disabled={Boolean(editingConversationId)}
                aria-invalid={Boolean(renameError)}
                aria-describedby={
                  renameError ? "rename-error" : "rename-hint"
                }
              />
              {renameError ? (
                <span
                  className="field-error"
                  id="rename-error"
                  role="alert"
                >
                  <i
                    className="bi bi-exclamation-circle"
                    aria-hidden="true"
                  ></i>
                  {renameError}
                </span>
              ) : (
                <span className="field-hint" id="rename-hint">
                  مئة حرف كحد أقصى.
                </span>
              )}
            </div>
          </form>
        </Modal>
      ) : null}

      {deleteTarget ? (
        <ConfirmDialog
          title="حذف المحادثة؟"
          description={`سيتم حذف «${conversationTitle(
            deleteTarget
          )}» وكل رسائلها. لا يمكن التراجع عن هذا الإجراء.`}
          confirmLabel="حذف المحادثة"
          cancelLabel="إلغاء"
          onConfirm={handleDeleteConversation}
          onClose={() => {
            if (!deleteBusy) setDeleteTarget(null);
          }}
          busy={deleteBusy}
          tone="danger"
          icon="bi-trash3"
        />
      ) : null}
    </main>
  );
}

export default Chat;
