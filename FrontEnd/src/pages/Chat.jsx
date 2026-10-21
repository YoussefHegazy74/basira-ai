
import { useCallback, useEffect, useRef, useState } from "react";
import api from "../services/api";
import { useAuth } from "../context/AuthContext";

function Chat() {
  const { user, authLoading } = useAuth();

  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [conversationId, setConversationId] = useState(null);
  const [conversations, setConversations] = useState([]);
  const [isOpeningConversation, setIsOpeningConversation] = useState(false);
  const [editingConversationId, setEditingConversationId] = useState(null);
  const [deletingConversationId, setDeletingConversationId] = useState(null);

  const textareaRef = useRef(null);
  const messagesEndRef = useRef(null);

  useEffect(function () {
    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "end",
    });
  }, [messages, isLoading]);

  // تحميل سجل المحادثات للمستخدم المسجل فقط
  const loadConversations = useCallback(async function () {
    if (!user) {
      setConversations([]);
      return;
    }

    try {
      const response = await api.get("/conversations?page=1&limit=10");
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
    }
  }, [user]);

  useEffect(function () {
    if (authLoading) return;

    if (user) {
      loadConversations();
    } else {
      setConversations([]);
      setConversationId(null);
      setMessages([]);
    }
  }, [authLoading, user, loadConversations]);

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

      setConversationId(data.conversation._id);

      setMessages(
        (data.messages || []).map(function (item, index) {
          return {
            id: item._id || `${item.role}-${index}`,
            text: item.content,
            sender: item.role,
            sources: item.sources || [],
          };
        })
      );

      setMessage("");
    } catch (requestError) {
      setError(
        requestError.response?.data?.message ||
          requestError.message ||
          "تعذر فتح المحادثة."
      );
    } finally {
      setIsOpeningConversation(false);
    }
  }

  // تغيير اسم المحادثة
  async function handleRenameConversation(item) {
    if (
      !user ||
      isLoading ||
      isOpeningConversation ||
      editingConversationId ||
      deletingConversationId
    ) {
      return;
    }

    const currentTitle = item.title || "محادثة جديدة";

    const newTitle = window.prompt(
      "اكتب الاسم الجديد للمحادثة:",
      currentTitle
    );

    // المستخدم ألغى العملية
    if (newTitle === null) {
      return;
    }

    const trimmedTitle = newTitle.trim();

    if (!trimmedTitle) {
      window.alert("اسم المحادثة لا يمكن أن يكون فارغًا.");
      return;
    }

    if (trimmedTitle === currentTitle) {
      return;
    }

    setError("");
    setEditingConversationId(item._id);

    try {
      await api.put(`/conversations/${item._id}`, {
        newTitle: trimmedTitle,
      });

      // تحديث الاسم في القائمة مباشرة
      setConversations(function (currentConversations) {
        return currentConversations.map(function (conversation) {
          if (String(conversation._id) === String(item._id)) {
            return {
              ...conversation,
              title: trimmedTitle,
            };
          }

          return conversation;
        });
      });

      await loadConversations();
    } catch (requestError) {
      console.error(
        "Failed to rename conversation:",
        requestError.response?.data || requestError.message
      );

      setError(
        requestError.response?.data?.message ||
          "تعذر تغيير اسم المحادثة. حاول مرة أخرى."
      );
    } finally {
      setEditingConversationId(null);
    }
  }

  // حذف المحادثة ورسائلها من السجل
  async function handleDeleteConversation(item) {
    if (
      !user ||
      isLoading ||
      isOpeningConversation ||
      editingConversationId ||
      deletingConversationId
    ) {
      return;
    }

    const confirmed = window.confirm(
      `هل أنت متأكد من حذف المحادثة "${item.title || "محادثة جديدة"}"؟\nسيتم حذف المحادثة ورسائلها، ولا يمكن التراجع عن ذلك.`
    );

    if (!confirmed) {
      return;
    }

    setError("");
    setDeletingConversationId(item._id);

    try {
      await api.delete(`/conversations/${item._id}`);

      // إزالة المحادثة المحذوفة من القائمة
      setConversations(function (currentConversations) {
        return currentConversations.filter(function (conversation) {
          return String(conversation._id) !== String(item._id);
        });
      });

      // لو المحادثة المحذوفة هي المفتوحة، ابدأ محادثة جديدة
      if (String(conversationId) === String(item._id)) {
        setConversationId(null);
        setMessages([]);
        setMessage("");
      }

      await loadConversations();
    } catch (requestError) {
      console.error(
        "Failed to delete conversation:",
        requestError.response?.data || requestError.message
      );

      setError(
        requestError.response?.data?.message ||
          "تعذر حذف المحادثة. حاول مرة أخرى."
      );
    } finally {
      setDeletingConversationId(null);
    }
  }

  async function handleSubmit(event) {
    event.preventDefault();

    const text = message.trim();

    if (!text || isLoading || isOpeningConversation) return;

    // حد الرسالة الواحدة للـ Guest هو 250 حرفًا،
    // وللمستخدم المسجل 500 حرف.
    const maxLength = user ? 500 : 250;

    if (text.length > maxLength) {
      setError(`الحد الأقصى لطول الرسالة هو ${maxLength} حرفًا.`);
      return;
    }

    setError("");
    setIsLoading(true);
    setMessage("");

    const previousMessages = messages;

    const userMessage = {
      id: `local-user-${Date.now()}`,
      text,
      sender: "user",
    };

    setMessages(function (currentMessages) {
      return [...currentMessages, userMessage];
    });

    try {
      // تجهيز آخر أربع رسائل سابقة للـ Guest.
      // السؤال الحالي لا يُضاف إلى chat_history.
      const chatHistory = previousMessages
        .filter(function (item) {
          return (
            (item.sender === "user" ||
              item.sender === "assistant") &&
            typeof item.text === "string" &&
            item.text.trim().length > 0
          );
        })
        .slice(-4)
        .map(function (item) {
          return {
            role: item.sender,
            content: item.text.trim(),
          };
        });

      const requestBody = {
        content: text,
        chat_history: chatHistory,
      };

      // إضافة معرف المحادثة للمستخدم المسجل فقط
      if (user && conversationId) {
        requestBody.conversationId = conversationId;
      }

      const response = await api.post("/messages", requestBody);

      const result = response.data || {};
      const data = result.data || {};

      if (user && data.conversation?._id) {
        setConversationId(data.conversation._id);
      }

      const answer =
        data.answer || data.assistantMessage?.content;

      if (typeof answer !== "string" || !answer.trim()) {
        throw new Error("لم يصل رد صالح من الخادم.");
      }

      const sources = Array.isArray(data.sources)
        ? data.sources
        : Array.isArray(data.assistantMessage?.sources)
          ? data.assistantMessage.sources
          : [];

      setMessages(function (currentMessages) {
        return [
          ...currentMessages,
          {
            id: `local-assistant-${Date.now()}`,
            text: answer,
            sender: "assistant",
            sources,
          },
        ];
      });

      // تحديث سجل المحادثات للمستخدم المسجل فقط
      if (user) {
        await loadConversations();
      }
    } catch (requestError) {
      console.error(
        "Failed to send message:",
        requestError.response?.data || requestError.message
      );

      setError(
        requestError.response?.data?.message ||
          requestError.response?.data?.error?.message ||
          requestError.message ||
          "تعذر الاتصال بالخادم. تأكد من تشغيل الـ Backend وخدمة الذكاء الاصطناعي."
      );
    } finally {
      setIsLoading(false);
      textareaRef.current?.focus();
    }
  }

  function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
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

    textareaRef.current?.focus();
  }

  function renderSource(source, index) {
    const sourceText =
      typeof source === "string"
        ? source
        : source?.title || source?.url || "";

    if (!sourceText) return null;

    const sourceUrl =
      typeof source === "object" ? source.url : sourceText;

    const isUrl =
      typeof sourceUrl === "string" &&
      /^https?:\/\//i.test(sourceUrl);

    return (
      <li key={`${sourceText}-${index}`}>
        {isUrl ? (
          <a
            href={sourceUrl}
            target="_blank"
            rel="noopener noreferrer"
          >
            {sourceText}
          </a>
        ) : (
          sourceText
        )}
      </li>
    );
  }

  return (
    <main className="chat-page">
      <div className="container chat-container">
        <div className="chat-page-heading">
          <h1>المساعد الذكي</h1>
          <p>اكتب سؤالك وابدأ محادثتك باللغة العربية.</p>
        </div>

        <div className="chat-layout">
          {/* سجل المحادثات للمستخدم المسجل */}
          {user && (
            <aside className="chat-sidebar">
              <div className="chat-sidebar-heading">
                <h3>سجل المحادثات</h3>
                <span>{conversations.length} / 10</span>
              </div>

              <button
                className="chat-new-button"
                type="button"
                onClick={handleNewChat}
                disabled={
                  isLoading ||
                  isOpeningConversation ||
                  Boolean(editingConversationId) ||
                  Boolean(deletingConversationId)
                }
              >
                <i className="bi bi-plus-lg"></i>
                دردشة جديدة
              </button>

              <div className="chat-sidebar-list">
                {conversations.length === 0 ? (
                  <p className="chat-sidebar-empty">
                    لا توجد محادثات محفوظة حتى الآن.
                  </p>
                ) : (
                  conversations.map(function (item) {
                    const isActive =
                      String(conversationId) === String(item._id);

                    const isRenaming =
                      String(editingConversationId) ===
                      String(item._id);

                    const isDeleting =
                      String(deletingConversationId) ===
                      String(item._id);

                    return (
                      <div
                        className={`chat-history-row ${
                          isActive ? "active" : ""
                        }`}
                        key={item._id}
                      >
                        {/* الضغط على اسم المحادثة يفتحها */}
                        <button
                          type="button"
                          className={`chat-history-item ${
                            isActive ? "active" : ""
                          }`}
                          onClick={function () {
                            openConversation(item);
                          }}
                          disabled={
                            isLoading ||
                            isOpeningConversation ||
                            Boolean(editingConversationId) ||
                            Boolean(deletingConversationId)
                          }
                          title={item.title || "محادثة جديدة"}
                        >
                          <i className="bi bi-chat-left-text"></i>

                          <span>
                            {item.title || "محادثة جديدة"}
                          </span>
                        </button>

                        {/* تغيير اسم المحادثة */}
                        <button
                          type="button"
                          className="chat-history-action rename"
                          title="تغيير اسم المحادثة"
                          aria-label="تغيير اسم المحادثة"
                          onClick={function () {
                            handleRenameConversation(item);
                          }}
                          disabled={
                            isLoading ||
                            isOpeningConversation ||
                            Boolean(editingConversationId) ||
                            Boolean(deletingConversationId)
                          }
                        >
                          <i
                            className={
                              isRenaming
                                ? "bi bi-hourglass-split"
                                : "bi bi-pencil-square"
                            }
                          ></i>
                        </button>

                        {/* حذف المحادثة */}
                        <button
                          type="button"
                          className="chat-history-action delete"
                          title="حذف المحادثة"
                          aria-label="حذف المحادثة"
                          onClick={function () {
                            handleDeleteConversation(item);
                          }}
                          disabled={
                            isLoading ||
                            isOpeningConversation ||
                            Boolean(editingConversationId) ||
                            Boolean(deletingConversationId)
                          }
                        >
                          <i
                            className={
                              isDeleting
                                ? "bi bi-hourglass-split"
                                : "bi bi-trash3"
                            }
                          ></i>
                        </button>
                      </div>
                    );
                  })
                )}
              </div>
            </aside>
          )}

          <section className="chat-workspace">
            <header className="chat-header">
              <div className="chat-header-info">
                <div className="chat-avatar">
                  <i className="bi bi-eye-fill"></i>
                </div>

                <div>
                  <h2>بصيرة AI</h2>

                  <span className="chat-status">
                    <span className="status-dot"></span>

                    {isOpeningConversation
                      ? "جاري تحميل المحادثة..."
                      : isLoading
                        ? "بصيرة تكتب الرد..."
                        : "جاهز للمحادثة"}
                  </span>
                </div>
              </div>

              {!user && (
                <button
                  className="chat-new-button"
                  type="button"
                  onClick={handleNewChat}
                  disabled={isLoading || isOpeningConversation}
                >
                  <i className="bi bi-plus-lg"></i>
                  دردشة جديدة
                </button>
              )}
            </header>

            <div className="chat-messages">
              {messages.length === 0 ? (
                <div className="chat-empty-state">
                  <div className="chat-empty-icon">
                    <i className="bi bi-chat-square-text"></i>
                  </div>

                  <h3>أهلًا بك في بصيرة</h3>
                  <p>
                    اكتب سؤالك في الأسفل لبدء محادثة جديدة.
                  </p>
                </div>
              ) : (
                <div className="message-list">
                  {messages.map(function (item) {
                    return (
                      <div
                        className={`message-row ${item.sender}`}
                        key={item.id}
                      >
                        {item.sender === "assistant" && (
                          <div className="message-avatar">
                            <i className="bi bi-eye-fill"></i>
                          </div>
                        )}

                        <div className="message-bubble">
                          <span className="message-sender">
                            {item.sender === "user"
                              ? "أنت"
                              : "بصيرة AI"}
                          </span>

                          <p style={{ whiteSpace: "pre-wrap" }}>
                            {item.text}
                          </p>

                          {item.sender === "assistant" &&
                            item.sources?.length > 0 && (
                              <div className="message-sources">
                                <strong>المصادر:</strong>
                                <ul>
                                  {item.sources.map(renderSource)}
                                </ul>
                              </div>
                            )}
                        </div>
                      </div>
                    );
                  })}

                  {isLoading && (
                    <div className="message-row assistant">
                      <div className="message-avatar">
                        <i className="bi bi-eye-fill"></i>
                      </div>

                      <div className="message-bubble">
                        <span className="message-sender">
                          بصيرة AI
                        </span>
                        <p>جاري التفكير في الإجابة...</p>
                      </div>
                    </div>
                  )}

                  <div ref={messagesEndRef} />
                </div>
              )}
            </div>

            {error && (
              <div
                className="chat-notice"
                role="alert"
                style={{
                  margin: "12px 20px",
                  color: "#b42318",
                }}
              >
                <i className="bi bi-exclamation-circle"></i>
                <span>{error}</span>
              </div>
            )}

            <form
              className="chat-composer"
              onSubmit={handleSubmit}
            >
              <label
                className="composer-label"
                htmlFor="chat-message"
              >
                رسالتك
              </label>

              <textarea
                id="chat-message"
                ref={textareaRef}
                value={message}
                onChange={function (event) {
                  setMessage(event.target.value);
                  if (error) setError("");
                }}
                onKeyDown={handleKeyDown}
                placeholder="اكتب سؤالك هنا..."
                rows={3}
                maxLength={user ? 500 : 250}
                disabled={isLoading || isOpeningConversation}
              />

              <div className="composer-footer">
                <button
                  className="primary-button send-message-button"
                  type="submit"
                  disabled={
                    !message.trim() ||
                    isLoading ||
                    isOpeningConversation
                  }
                >
                  {isLoading ? "جاري الإرسال..." : "إرسال"}

                  <i
                    className={
                      isLoading
                        ? "bi bi-hourglass-split"
                        : "bi bi-send"
                    }
                  ></i>
                </button>
              </div>
            </form>
          </section>
        </div>
      </div>
    </main>
  );
}

export default Chat;
