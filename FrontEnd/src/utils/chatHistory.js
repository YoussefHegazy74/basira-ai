const ACTIVE_CONVERSATION_KEY = "basira.activeConversation";
const FAILED_SUBMISSION_KEY = "basira.failedSubmission";

export function activeConversationKey(user) {
  const identity = user?.id || user?._id || user?.email;
  return identity ? `${ACTIVE_CONVERSATION_KEY}.${identity}` : null;
}

export function readActiveConversation(user, storage = globalThis.localStorage) {
  const key = activeConversationKey(user);
  if (!key || !storage) return null;
  try {
    return storage.getItem(key);
  } catch {
    return null;
  }
}

export function writeActiveConversation(
  user,
  conversationId,
  storage = globalThis.localStorage,
) {
  const key = activeConversationKey(user);
  if (!key || !storage) return;
  try {
    if (conversationId) storage.setItem(key, String(conversationId));
    else storage.removeItem(key);
  } catch {
    /* storage unavailable */
  }
}

function failedSubmissionKey(user) {
  const identity = user?.id || user?._id || user?.email;
  return identity ? `${FAILED_SUBMISSION_KEY}.${identity}` : null;
}

export function readFailedSubmission(user, storage = globalThis.localStorage) {
  const key = failedSubmissionKey(user);
  if (!key || !storage) return null;
  try {
    const raw = storage.getItem(key);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (
      !parsed ||
      typeof parsed.text !== "string" ||
      typeof parsed.clientMessageId !== "string"
    ) {
      return null;
    }
    return parsed;
  } catch {
    return null;
  }
}

export function writeFailedSubmission(
  user,
  failed,
  storage = globalThis.localStorage,
) {
  const key = failedSubmissionKey(user);
  if (!key || !storage) return;
  try {
    if (failed) storage.setItem(key, JSON.stringify(failed));
    else storage.removeItem(key);
  } catch {
    /* storage unavailable */
  }
}

export function buildChatHistory(messages, excludeClientId) {
  return (Array.isArray(messages) ? messages : [])
    .filter(function (item) {
      return (
        (item.sender === "user" || item.sender === "assistant") &&
        typeof item.text === "string" &&
        item.text.trim().length > 0 &&
        (!excludeClientId || item.clientMessageId !== excludeClientId)
      );
    })
    .slice(-4)
    .map(function (item) {
      return {
        role: item.sender,
        content: item.text.trim(),
      };
    });
}

export function mapStoredMessages(messages) {
  return (Array.isArray(messages) ? messages : []).map(function (item, index) {
    return {
      id: item._id || `${item.role}-${index}`,
      text: item.content,
      sender: item.role,
      sources: item.sources || [],
    };
  });
}

export function makeClientMessageId() {
  const cryptoApi = globalThis.crypto;
  if (cryptoApi && typeof cryptoApi.randomUUID === "function") {
    return cryptoApi.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

export function mergeServerMessages(
  currentMessages,
  data,
  clientMessageId,
  { appendUser = false } = {},
) {
  const answer = data.answer || data.assistantMessage?.content;
  if (typeof answer !== "string" || !answer.trim()) {
    throw new Error("لم يصل رد صالح من الخادم.");
  }

  const sources = Array.isArray(data.sources)
    ? data.sources
    : Array.isArray(data.assistantMessage?.sources)
      ? data.assistantMessage.sources
      : [];
  const serverUserId = data.userMessage?._id;
  const assistantId =
    data.assistantMessage?._id || `server-assistant-${Date.now()}`;

  const next = (Array.isArray(currentMessages) ? currentMessages : []).map(
    function (item) {
      if (item.clientMessageId === clientMessageId && serverUserId) {
        return { ...item, id: serverUserId };
      }
      return item;
    },
  );

  if (
    appendUser &&
    data.userMessage &&
    !next.some((item) => item.id === serverUserId)
  ) {
    next.push({
      id: serverUserId,
      text: data.userMessage.content || "",
      sender: "user",
      clientMessageId,
    });
  }

  if (!next.some((item) => item.id === assistantId)) {
    next.push({
      id: assistantId,
      text: answer,
      sender: "assistant",
      sources,
    });
  }

  return next;
}
