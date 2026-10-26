const mongoose = require("mongoose");
const Message = require("../message/message.model.js");
const conversationService = require("../conversation/conversation.service.js");
const UsageService = require("../usage/usage.service.js");
const AppError = require("../utils/AppError.js");
const HttpStatus = require("../utils/HttpStatusText.js");

/// constants
const HISTORY_LIMIT = 4;
const MAX_CONTENT_LENGTH = 500;
const GUEST_MAX_CONTENT_LENGTH = 250;
const MAX_MESSAGES_PER_CONVERSATION = 10;

/// last N messages BEFORE the current one, oldest -> newest
const buildHistory = async (conversationId, beforeMessage = null) => {
  const filter = { conversationId };
  if (beforeMessage) {
    // Exclude the current user message and anything created after it.  The
    // fallback on _id makes the boundary deterministic even when two
    // messages share the same millisecond.
    filter.$or = [
      { createdAt: { $lt: beforeMessage.createdAt } },
      {
        createdAt: beforeMessage.createdAt,
        _id: { $lt: beforeMessage._id },
      },
    ];
  }
  const recent = await Message.find(filter)
    .sort({ createdAt: -1, _id: -1 })
    .limit(HISTORY_LIMIT)
    .select("role content -_id")
    .lean();
  return recent.reverse();
};

const normalizeClientMessageId = (value) => {
  if (value === undefined || value === null || value === "") return null;
  if (typeof value !== "string") {
    throw new AppError(400, "clientMessageId must be a string", HttpStatus.BAD_REQUEST);
  }
  const trimmed = value.trim();
  if (!trimmed || trimmed.length > 128) {
    throw new AppError(400, "Invalid clientMessageId", HttpStatus.BAD_REQUEST);
  }
  return trimmed;
};

const findExistingUserMessage = async (conversationId, clientMessageId) => {
  if (!clientMessageId) return null;
  return Message.findOne({
    conversationId,
    role: "user",
    clientMessageId,
  });
};

const findExistingReply = async (userMessage) => {
  if (!userMessage) return null;
  const linked = await Message.findOne({
    conversationId: userMessage.conversationId,
    role: "assistant",
    replyTo: userMessage._id,
  });
  if (linked) return linked;

  // Legacy messages have no replyTo.  Only the immediately following
  // message can be this user message's reply; a later assistant belongs to
  // a subsequent turn and must not be returned for an old retry.
  const next = await Message.findOne({
    conversationId: userMessage.conversationId,
    $or: [
      { createdAt: { $gt: userMessage.createdAt } },
      {
        createdAt: userMessage.createdAt,
        _id: { $gt: userMessage._id },
      },
    ],
  }).sort({ createdAt: 1, _id: 1 });
  return next && next.role === "assistant" ? next : null;
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
      throw new Error(`استجاب خادم الذكاء الاصطناعي بالرمز ${response.status}`);
    }

    const data = await response.json();
    return { answer: data.answer, sources: data.sources || [] };
  } catch (error) {
    console.error("AI model failed:", error.message);
    throw new AppError(502, "فشل نموذج الذكاء الاصطناعي", HttpStatus.ERROR);
  }
};

/// never trust what a guest sends
/// returns { query, chatHistory }
const sanitizeGuestInput = (message, chatHistory) => {
  if (typeof message !== "string" || !message.trim()) {
    throw new AppError(400, "الرسالة مطلوبة", HttpStatus.BAD_REQUEST);
  }
  const query = message.trim();
  if (query.length > GUEST_MAX_CONTENT_LENGTH) {
    throw new AppError(
      400,
      `الرسالة تتجاوز الحد الأقصى للطول البالغ ${GUEST_MAX_CONTENT_LENGTH} حرفًا`,
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
  const { ownerKey } = UsageService.identify(null, guestKey);
  await UsageService.consume(ownerKey);
  const { answer, sources } = await askModel(query, cleanHistory);
  return { answer, sources };
};

/// logged in user -> everything is saved in db
const sendUserMessage = async ({
  userId,
  conversationId,
  content,
  clientMessageId,
}) => {
  if (typeof content !== "string" || !content.trim()) {
    throw new AppError(400, "الرسالة مطلوبة", HttpStatus.BAD_REQUEST);
  }
  content = content.trim();
  if (content.length > MAX_CONTENT_LENGTH) {
    throw new AppError(
      400,
      `الرسالة تتجاوز الحد الأقصى للطول البالغ ${MAX_CONTENT_LENGTH} حرفًا`,
      HttpStatus.BAD_REQUEST,
    );
  }
  const idempotencyKey = normalizeClientMessageId(clientMessageId);

  // Ownership is validated before any usage or write.  An invalid/stale
  // conversationId must never consume quota or create a second conversation.
  let conversation = conversationId
    ? await conversationService.getOwnedOrThrow(conversationId, userId)
    : null;

  let userMessage = null;
  if (idempotencyKey) {
    // A lost response can leave the client without the new conversation ID.
    // Look the key up globally, then verify that the owning conversation
    // belongs to this user before reusing it.
    const existing = await Message.findOne({
      role: "user",
      clientMessageId: idempotencyKey,
    });
    if (existing) {
      const existingConversation = await conversationService.getOwnedOrThrow(
        existing.conversationId,
        userId,
      );
      if (
        conversation &&
        String(conversation._id) !== String(existingConversation._id)
      ) {
        throw new AppError(
          409,
          "clientMessageId already belongs to another conversation",
          HttpStatus.FAIL,
        );
      }
      conversation = existingConversation;
      userMessage = existing;
    }
  }
  if (!userMessage && conversation) {
    userMessage = await findExistingUserMessage(
      conversation._id,
      idempotencyKey,
    );
  }

  if (!userMessage) {
    let createdConversation = false;
    if (!conversation) {
      conversation = await conversationService.createConversation(userId, content);
      createdConversation = true;
    }
    // A new turn needs room for both the user message and its reply.  The
    // counter is cumulative by design (deleting does not refund it).
    const neededSlots = 2;
    if (
      conversation.messageCount + neededSlots >
      MAX_MESSAGES_PER_CONVERSATION
    ) {
      throw new AppError(
        403,
        "تم الوصول إلى الحد الأقصى لعدد الرسائل في هذه المحادثة. "
          + "ابدأ محادثة جديدة للمتابعة.",
        HttpStatus.FAIL,
      );
    }

    const { ownerKey } = UsageService.identify(userId, null);
    await UsageService.consume(ownerKey);

    try {
      userMessage = await Message.create({
        conversationId: conversation._id,
        role: "user",
        content,
        clientMessageId: idempotencyKey,
      });
      await conversationService.incrementMessageCount(conversation._id, 1);
    } catch (error) {
      // Concurrent retries with the same key race on the unique index.  The
      // loser reuses the winner's persisted user message and its owning
      // conversation; any empty conversation it created is removed.
      if (error.code === 11000 && idempotencyKey) {
        const winner = await Message.findOne({
          role: "user",
          clientMessageId: idempotencyKey,
        });
        if (winner) {
          const winnerConversation = await conversationService.getOwnedOrThrow(
            winner.conversationId,
            userId,
          );
          if (createdConversation) {
            await conversationService.deleteIfEmpty(
              conversation._id,
              userId,
            );
          }
          conversation = winnerConversation;
          userMessage = winner;
        }
      }
      if (!userMessage) {
        if (createdConversation) {
          await conversationService.deleteIfEmpty(conversation._id, userId);
        }
        throw error;
      }
    }
  }

  const existingReply = await findExistingReply(userMessage);
  if (existingReply) {
    return {
      conversation,
      userMessage,
      assistantMessage: existingReply,
      replayed: true,
    };
  }

  // Only history BEFORE this user message is sent to the model.  The current
  // question is passed separately as `query`, so it is never duplicated.
  const chatHistory = await buildHistory(conversation._id, userMessage);

  let result;
  try {
    result = await askModel(content, chatHistory);
  } catch (error) {
    // Keep the user message for a safe retry and give the client the IDs it
    // needs to reconcile the optimistic bubble.  Without these IDs the
    // frontend could neither reopen nor retry the saved conversation.
    error.data = {
      conversationId: String(conversation._id),
      userMessageId: String(userMessage._id),
      clientMessageId: idempotencyKey,
      retryable: true,
    };
    throw error;
  }

  let assistantMessage;
  try {
    assistantMessage = await Message.create({
      conversationId: conversation._id,
      role: "assistant",
      content: result.answer,
      sources: result.sources,
      replyTo: userMessage._id,
    });
    await conversationService.incrementMessageCount(conversation._id, 1);
  } catch (error) {
    // A concurrent retry may have already persisted the reply.  Reuse it
    // instead of failing a request that actually has a valid answer.
    if (error.code === 11000) {
      assistantMessage = await findExistingReply(userMessage);
    }
    if (!assistantMessage) throw error;
  }

  return {
    conversation,
    userMessage,
    assistantMessage,
    replayed: false,
  };
};

/// entry point: no userId -> guest
const sendMessage = async ({
  userId,
  conversationId,
  content,
  chatHistory,
  guestKey,
  clientMessageId,
}) => {
  if (!userId) return sendGuestMessage(content, chatHistory, guestKey);
  return sendUserMessage({
    userId,
    conversationId,
    content,
    clientMessageId,
  });
};

/// delete one message (and the AI reply after it if it's a user message)
/// messageCount is NOT decremented, so deleting can't bypass the limit
const deleteMessage = async ({ messageId, userId }) => {
  if (!mongoose.isValidObjectId(messageId)) {
    throw new AppError(404, "الرسالة غير موجودة", HttpStatus.FAIL);
  }
  const message = await Message.findById(messageId);
  if (!message) {
    throw new AppError(404, "الرسالة غير موجودة", HttpStatus.FAIL);
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
