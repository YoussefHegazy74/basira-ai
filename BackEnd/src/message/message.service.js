const mongoose = require("mongoose");
const Message = require("../message/message.model.js");
const conversationService = require("../conversation/conversation.service.js");
const UsageService = require("../usage/usage.service.js");
const AppError = require("../utils/appError.js");
const HttpStatus = require("../utils/HttpStatusText.js");

/// constants
const HISTORY_LIMIT = 4;
const MAX_CONTENT_LENGTH = 500;
const GUEST_MAX_CONTENT_LENGTH = 250;
const MAX_MESSAGES_PER_CONVERSATION = 10;

/// last N messages BEFORE the current one, oldest -> newest
const buildHistory = async (conversationId) => {
  const recent = await Message.find({ conversationId })
    .sort({ createdAt: -1 })
    .limit(HISTORY_LIMIT)
    .select("role content -_id")
    .lean();
  return recent.reverse();
};

/// call the model: query = current question, chatHistory = previous messages
const askModel = async (query, chatHistory = []) => {
  try {
    const response = await fetch(`${process.env.AI_URL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, chat_history: chatHistory }),
      signal: AbortSignal.timeout(60000),
    });

    if (!response.ok) {
      throw new Error(`AI server responded with ${response.status}`);
    }

    const data = await response.json();
    return { answer: data.answer, sources: data.sources || [] };
  } catch (error) {
    console.error("AI model failed:", error.message);
    throw new AppError(502, "AI model failed", HttpStatus.ERROR);
  }
};

/// never trust what a guest sends
/// returns { query, chatHistory }
const sanitizeGuestInput = (message, chatHistory) => {
  if (typeof message !== "string" || !message.trim()) {
    throw new AppError(400, "Message is required", HttpStatus.BAD_REQUEST);
  }
  const query = message.trim();
  if (query.length > GUEST_MAX_CONTENT_LENGTH) {
    throw new AppError(
      400,
      `Message exceeds maximum length of ${GUEST_MAX_CONTENT_LENGTH} characters`,
      HttpStatus.BAD_REQUEST,
    );
  }
  const history = Array.isArray(chatHistory) ? chatHistory : [];
  const cleanHistory = history
    .filter(
      (m) =>
        m &&
        ["user", "assistant"].includes(m.role) &&
        typeof m.content === "string" &&
        m.content.trim(),
    )
    .slice(-HISTORY_LIMIT)
    .map((m) => ({
      role: m.role,
      content: m.content.trim().slice(0, GUEST_MAX_CONTENT_LENGTH),
    }));

  return { query, cleanHistory };
};

/// guest -> history comes from the request, nothing is saved in db
const sendGuestMessage = async (message, chatHistory, guestKey) => {
  const { query, cleanHistory } = sanitizeGuestInput(message, chatHistory);
  const { ownerKey, limit } = identify(null, guestKey);
  await consume(ownerKey, limit);
  const { answer, sources } = await askModel(query, cleanHistory);
  return { answer, sources };
};

/// logged in user -> everything is saved in db
const sendUserMessage = async ({ userId, conversationId, content }) => {
  if (typeof content !== "string" || !content.trim()) {
    throw new AppError(400, "Message is required", HttpStatus.BAD_REQUEST);
  }
  content = content.trim();
  if (content.length > MAX_CONTENT_LENGTH) {
    throw new AppError(
      400,
      `Message exceeds maximum length of ${MAX_CONTENT_LENGTH} characters`,
      HttpStatus.BAD_REQUEST,
    );
  }
  const { ownerKey, limit } = identify(userId);
  await consume(ownerKey, limit);
  const conversation = conversationId
    ? await conversationService.getOwnedOrThrow(conversationId, userId)
    : await conversationService.createConversation(userId, content);
  // limit counts every message ever sent (delete does not decrement it)
  if (conversation.messageCount >= MAX_MESSAGES_PER_CONVERSATION) {
    throw new AppError(
      403,
      "Conversation message limit reached, please start a new chat",
      HttpStatus.FAIL,
    );
  }

  const chatHistory = await buildHistory(conversation._id);

  const userMessage = await Message.create({
    conversationId: conversation._id,
    role: "user",
    content,
  });

  let result;
  try {
    result = await askModel(content, chatHistory);
  } catch (error) {
    // user message stays saved so the frontend can retry
    await conversationService.incrementMessageCount(conversation._id, 1);
    throw error;
  }

  const assistantMessage = await Message.create({
    conversationId: conversation._id,
    role: "assistant",
    content: result.answer,
    sources: result.sources,
  });
  await conversationService.incrementMessageCount(conversation._id, 2);

  return { conversation, userMessage, assistantMessage };
};

/// entry point: no userId -> guest
const sendMessage = async ({
  userId,
  conversationId,
  content,
  chatHistory,
  guestKey,
}) => {
  if (!userId) return sendGuestMessage(content, chatHistory, guestKey);
  return sendUserMessage({ userId, conversationId, content });
};

/// delete one message (and the AI reply after it if it's a user message)
/// messageCount is NOT decremented, so deleting can't bypass the limit
const deleteMessage = async ({ messageId, userId }) => {
  if (!mongoose.isValidObjectId(messageId)) {
    throw new AppError(404, "Message not found", HttpStatus.FAIL);
  }
  const message = await Message.findById(messageId);
  if (!message) {
    throw new AppError(404, "Message not found", HttpStatus.FAIL);
  }
  // ownership is checked through the conversation
  await conversationService.getOwnedOrThrow(message.conversationId, userId);

  const ids = [message._id];

  if (message.role === "user") {
    const next = await Message.findOne({
      conversationId: message.conversationId,
      createdAt: { $gt: message.createdAt },
    }).sort({ createdAt: 1 });
    if (next && next.role === "assistant") ids.push(next._id);
  }

  await Message.deleteMany({ _id: { $in: ids } });
};

module.exports = { sendMessage, deleteMessage };
