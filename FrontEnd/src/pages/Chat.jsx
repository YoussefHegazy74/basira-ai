import { useCallback, useEffect, useRef, useState } from "react";
import { useLocation } from "react-router-dom";
import api from "../services/api";
import { useAuth } from "../context/AuthContext";
import { useChatNavigation } from "../context/ChatNavigationContext";
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
import { citationSegments, formatSourceLabel } from "../utils/citations";
import { arabicMessage, formatNumber } from "../utils/format";
import Modal, { ConfirmDialog } from "../components/Modal";
import "../styles/chat.css";

function sourceLabelsFor(item) {
  return (Array.isArray(item?.sources) ? item.sources : [])
    .map(formatSourceLabel)
    .filter(Boolean);
}

function conversationTitle(item) {
  return item?.title || "محادثة جديدة";
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
  const [feedback, setFeedback] = useState({});
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
  const newChatTokenRef = useRef(null);

  const location = useLocation();
  const { setHistory, setActions } = useChatNavigation();
  const userIdentity = user?.id || user?._id || user?.email || null;
  const maxLength = user ? 500 : 250;
  const hasMessages = messages.length > 0;

  // الحفاظ على موضع الكتابة أثناء نمو الحقل.
  useEffect(
    function () {
      const field = textareaRef.current;
      if (!field) return;

      field.style.height = "auto";
      const nextHeight = Math.min(field.scrollHeight, 176);
      field.style.height = `${nextHeight}px`;
      field.style.overflowY =
        field.scrollHeight > 176 ? "auto" : "hidden";
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
      if (location.state?.newChat) return;
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

  async function copyText(text, key) {
    if (!text) return;

    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(text);
      } else {
        fallbackCopy(text);
      }

      setCopiedId(key);
      if (copyTimerRef.current) {
        window.clearTimeout(copyTimerRef.current);
      }
      copyTimerRef.current = window.setTimeout(function () {
        setCopiedId("");
      }, 2000);
    } catch {
      setError("تعذّر النسخ من المتصفح. حدّد النص وانسخه يدويًا.");
    }
  }

  function handleCopy(item) {
    return copyText(item?.text || "", `message:${item.id}`);
  }

  function handleCopySources(item) {
    return copyText(sourceLabelsFor(item).join("\n"), `sources:${item.id}`);
  }

  function handleFeedback(item, value) {
    setFeedback(function (current) {
      return {
        ...current,
        [item.id]: current[item.id] === value ? 0 : value,
      };
    });
  }

  function closeMessageMenu(event) {
    const menu = event.currentTarget.closest("details");
    if (menu) menu.open = false;
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
    setShowScrollDown(false);
    stickToBottomRef.current = true;
    writeActiveConversation(user, null);
    writeFailedSubmission(user, null);

    textareaRef.current?.focus();
  }

  useEffect(
    function () {
      setHistory({
        conversations,
        activeId: conversationId,
        loaded: historyLoaded,
        busy:
          isLoading ||
          isOpeningConversation ||
          Boolean(editingConversationId) ||
          Boolean(deletingConversationId),
      });
    },
    [
      conversations,
      conversationId,
      historyLoaded,
      isLoading,
      isOpeningConversation,
      editingConversationId,
      deletingConversationId,
      setHistory,
    ]
  );

  useEffect(function () {
    setActions({
      newChat: handleNewChat,
      open: openConversation,
      rename: openRenameDialog,
      remove: openDeleteDialog,
    });
  });

  useEffect(
    function () {
      const token = location.state?.newChat;
      if (!token || token === newChatTokenRef.current) return;
      newChatTokenRef.current = token;
      handleNewChat();
    },
    [location.state, handleNewChat]
  );

  const deleteBusy =
    Boolean(deleteTarget) &&
    String(deletingConversationId) === String(deleteTarget?._id);

  function renderMessageText(item) {
    return citationSegments(item.text, item.sources).map(function (
      segment,
      index
    ) {
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
        <span key={`${item.id}-text-${index}`}>{segment.text}</span>
      );
    });
  }

  const composer = (
    <form className="chat-composer" onSubmit={handleSubmit}>
      <div className={`composer-pill${message.trim() ? " has-text" : ""}`}>
        <button
          type="button"
          className="composer-icon"
          disabled
          aria-label="إرفاق ملف، غير متاح بعد"
          title="إرفاق ملف، غير متاح بعد"
        >
          <i className="bi bi-plus-lg" aria-hidden="true"></i>
        </button>

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
          placeholder="اسأل بصيرة…"
          rows={1}
          maxLength={maxLength}
          dir="auto"
          disabled={isOpeningConversation}
        />

        <div className="composer-actions">
          <button
            type="button"
            className="composer-icon"
            disabled
            aria-label="الإدخال الصوتي، غير متاح بعد"
            title="الإدخال الصوتي، غير متاح بعد"
          >
            <i className="bi bi-mic" aria-hidden="true"></i>
          </button>

          <button
            className={`composer-send${message.trim() ? " is-ready" : ""}`}
            type="submit"
            disabled={!message.trim() || isLoading || isOpeningConversation}
            aria-busy={isLoading}
            aria-label={isLoading ? "جارٍ الإرسال" : "إرسال الرسالة"}
            title={isLoading ? "جارٍ الإرسال" : "إرسال الرسالة"}
          >
            {isLoading ? (
              <span className="send-dot" aria-hidden="true"></span>
            ) : (
              <i className="bi bi-arrow-up" aria-hidden="true"></i>
            )}
          </button>
        </div>
      </div>
    </form>
  );

  return (
    <main className="chat-app" aria-label="المحادثة" tabIndex={-1}>
      <h1 className="sr-only">المحادثة</h1>
      <div
        className={`chat-stage ${hasMessages ? "has-thread" : "is-empty"}`}
      >
        {hasMessages ? (
          <div
            className="chat-thread"
            ref={messagesRef}
            onScroll={handleMessagesScroll}
            aria-label="الرسائل"
          >
            <div className="chat-thread-inner">
              <div className="message-list">
                {messages.map(function (item) {
                  const labels = sourceLabelsFor(item);
                  const isCopied = copiedId === `message:${item.id}`;
                  const isSourcesCopied =
                    copiedId === `sources:${item.id}`;
                  const rating = feedback[item.id] || 0;

                  if (item.sender === "user") {
                    return (
                      <article className="message-row user" key={item.id}>
                        <div className="user-message">
                          <p className="user-text" dir="auto">
                            {renderMessageText(item)}
                          </p>
                          <div className="user-message-foot">
                            <button
                              type="button"
                              className={`message-action${
                                isCopied ? " is-copied" : ""
                              }`}
                              onClick={() => handleCopy(item)}
                              aria-label={
                                isCopied
                                  ? "تم نسخ الرسالة"
                                  : "نسخ الرسالة"
                              }
                              title={
                                isCopied
                                  ? "تم نسخ الرسالة"
                                  : "نسخ الرسالة"
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
                        </div>
                      </article>
                    );
                  }

                  return (
                    <article
                      className="message-row assistant"
                      key={item.id}
                    >
                      <div className="assistant-message">
                        <p className="assistant-text" dir="auto">
                          {renderMessageText(item)}
                        </p>

                        {labels.length ? (
                          <div
                            className="message-sources"
                            aria-label="مصادر الإجابة"
                          >
                            <div className="source-list">
                              {labels.map((label, index) => (
                                <span
                                  className="source-chip"
                                  key={`${item.id}-source-${index}`}
                                >
                                  <i
                                    className="bi bi-journal-text"
                                    aria-hidden="true"
                                  ></i>
                                  {label}
                                </span>
                              ))}
                            </div>
                          </div>
                        ) : null}

                        <div
                          className="message-actions"
                          role="group"
                          aria-label="إجراءات الرسالة"
                        >
                          <button
                            type="button"
                            className={`message-action${
                              isCopied ? " is-copied" : ""
                            }`}
                            onClick={() => handleCopy(item)}
                            aria-label={
                              isCopied ? "تم نسخ الإجابة" : "نسخ الإجابة"
                            }
                            title={
                              isCopied ? "تم نسخ الإجابة" : "نسخ الإجابة"
                            }
                          >
                            <i
                              className={
                                isCopied ? "bi bi-check2" : "bi bi-copy"
                              }
                              aria-hidden="true"
                            ></i>
                          </button>

                          <button
                            type="button"
                            className="message-action is-disabled"
                            disabled
                            aria-label="إعادة التوليد غير متاحة في هذه النسخة"
                            title="إعادة التوليد غير متاحة في هذه النسخة"
                          >
                            <i
                              className="bi bi-arrow-clockwise"
                              aria-hidden="true"
                            ></i>
                          </button>

                          <button
                            type="button"
                            className={`message-action${
                              rating === 1 ? " is-active" : ""
                            }`}
                            onClick={() => handleFeedback(item, 1)}
                            aria-pressed={rating === 1}
                            aria-label="إعجاب بالإجابة"
                            title="إعجاب"
                          >
                            <i
                              className="bi bi-hand-thumbs-up"
                              aria-hidden="true"
                            ></i>
                          </button>

                          <button
                            type="button"
                            className={`message-action${
                              rating === -1 ? " is-active" : ""
                            }`}
                            onClick={() => handleFeedback(item, -1)}
                            aria-pressed={rating === -1}
                            aria-label="عدم الإعجاب بالإجابة"
                            title="عدم إعجاب"
                          >
                            <i
                              className="bi bi-hand-thumbs-down"
                              aria-hidden="true"
                            ></i>
                          </button>

                          <details className="message-more">
                            <summary
                              className="message-action"
                              aria-label="المزيد من الخيارات"
                              title="المزيد"
                            >
                              <i
                                className="bi bi-three-dots"
                                aria-hidden="true"
                              ></i>
                            </summary>
                            <div className="message-more-menu">
                              <button
                                type="button"
                                onClick={(event) => {
                                  handleCopy(item);
                                  closeMessageMenu(event);
                                }}
                              >
                                <i
                                  className="bi bi-copy"
                                  aria-hidden="true"
                                ></i>
                                نسخ النص
                              </button>
                              {labels.length ? (
                                <button
                                  type="button"
                                  onClick={(event) => {
                                    handleCopySources(item);
                                    closeMessageMenu(event);
                                  }}
                                >
                                  <i
                                    className="bi bi-journal-text"
                                    aria-hidden="true"
                                  ></i>
                                  {isSourcesCopied
                                    ? "تم نسخ المصادر"
                                    : "نسخ المصادر"}
                                </button>
                              ) : null}
                            </div>
                          </details>
                        </div>
                      </div>
                    </article>
                  );
                })}

                {isLoading ? (
                  <article className="message-row assistant typing-row">
                    <div className="typing-state" role="status">
                      <span className="typing-dots" aria-hidden="true">
                        <span></span>
                        <span></span>
                        <span></span>
                      </span>
                      <span className="sr-only">بصيرة تكتب الآن…</span>
                    </div>
                  </article>
                ) : null}
              </div>
            </div>
          </div>
        ) : isOpeningConversation ? (
          <div className="chat-loading" aria-live="polite">
            <span className="typing-dots" aria-hidden="true">
              <span></span>
              <span></span>
              <span></span>
            </span>
            <span className="sr-only">جارٍ تحميل المحادثة…</span>
          </div>
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
                    <span className="send-dot" aria-hidden="true"></span>
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

        <div className="chat-composer-dock">{composer}</div>

        {hasMessages && showScrollDown ? (
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
