const mongoose = require("mongoose");
const Conversation = require("../conversation/conversation.model.js");
const Message = require("../message/message.model.js");
const AppError = require("../utils/AppError.js");
const HttpStatus = require("../utils/HttpStatusText.js");

/// get conversation that belongs to this user (404 if not found or not his)
const getOwnedOrThrow = async (conversationId, userId) => {
  if (!mongoose.isValidObjectId(conversationId)) {
    throw new AppError(404, "Conversation not found", HttpStatus.FAIL);
  }
  const conversation = await Conversation.findOne({
    _id: conversationId,
    userId,
  });
  if (!conversation) {
    throw new AppError(404, "Conversation not found", HttpStatus.FAIL);
  }
  return conversation;
};

//// create new conversation
const createConversation = async (userId, firstMessage) => {
  const title = firstMessage ? firstMessage.trim().slice(0, 40) : "New chat";
  return await Conversation.create({
    userId,
    title,
  });
};

//////// list user conversations
const getUserConversations = async (userId, page, limit) => {
  const conversations = await Conversation.find({ userId })
    .sort({
      updatedAt: -1,
    })
    .skip((page - 1) * limit)
    .limit(limit)
    .lean();
  return conversations;
};
//// get conversation by id
const getConversationById = async (conversationId, userId) => {
  const conversation = await getOwnedOrThrow(conversationId, userId);
  const messages = await Message.find({ conversationId: conversation._id })
    .sort({ createdAt: 1 })
    .lean();
  return { conversation, messages };
};

/// rename conversation
const renameConversation = async ({ conversationId, userId, newTitle }) => {
  title = typeof newTitle === "string" ? newTitle.trim() : "";
  if (!title) {
    throw new AppError(400, "Title is required", HttpStatus.BAD_REQUEST);
  }
  const conversation = await Conversation.findOneAndUpdate(
    { _id: conversationId, userId },
    { title: title.slice(0, 100) },
    { returnDocument: "after", runValidators: true },
  );
  if (!conversation) {
    throw new AppError(404, "Conversation not found", HttpStatus.FAIL);
  }
  return conversation;
};

/// delete conversation with all its messages
const deleteConversation = async ({ conversationId, userId }) => {
  const conversation = await getOwnedOrThrow(conversationId, userId);
  await Message.deleteMany({ conversationId: conversation._id });
  await conversation.deleteOne();
};

//// increment message count for conversation
const incrementMessageCount = async (conversationId, by = 1) => {
  await Conversation.updateOne(
    { _id: conversationId },
    { $inc: { messageCount: by } },
  );
};

module.exports = {
  getOwnedOrThrow,
  createConversation,
  getUserConversations,
  getConversationById,
  renameConversation,
  deleteConversation,
  incrementMessageCount,
};
