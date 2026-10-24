const conversationService = require("./conversation.service");
const HttpStatus = require("../utils/HttpStatusText.js");
const asyncWrapper = require("../middleware/Async_wrapper.js");

// 1. Create a new conversation
const createConversation = asyncWrapper(async (req, res, next) => {
  const conversation = await conversationService.createConversation(
    req.user._id,
    req.body.message,
  );
  res.status(201).json({
    status: HttpStatus.SUCCESS,
    data: conversation,
  });
});

//  get user conversations
const getUserConversations = asyncWrapper(async (req, res, next) => {
  // Bound pagination so an authenticated client cannot request an
  // arbitrarily large page or negative skip.
  const parsedPage = Number.parseInt(req.query.page, 10);
  const parsedLimit = Number.parseInt(req.query.limit, 10);
  const page = Number.isFinite(parsedPage) && parsedPage > 0 ? parsedPage : 1;
  const limit = Number.isFinite(parsedLimit)
    ? Math.min(50, Math.max(1, parsedLimit))
    : 10;
  const conversations = await conversationService.getUserConversations(
    req.user._id,
    page,
    limit,
  );
  res.status(200).json({
    status: HttpStatus.SUCCESS,
    data: conversations,
  });
});
//// get conversation by id
const getConversationById = asyncWrapper(async (req, res, next) => {
  const conversation = await conversationService.getConversationById(
    req.params.conversationId,
    req.user._id,
  );
  res.status(200).json({
    status: HttpStatus.SUCCESS,
    data: conversation,
  });
});
/// rename conversation
const renameConversation = asyncWrapper(async (req, res, next) => {
  const conversation = await conversationService.renameConversation({
    conversationId: req.params.conversationId,
    userId: req.user._id,
    newTitle: req.body.newTitle,
  });
  res.status(200).json({
    status: HttpStatus.SUCCESS,
    data: conversation,
  });
});

//// delete conversation
const deleteConversation = asyncWrapper(async (req, res, next) => {
  await conversationService.deleteConversation({
    conversationId: req.params.conversationId,
    userId: req.user._id,
  });
  res.status(200).json({
    status: HttpStatus.SUCCESS,
    data: null,
  });
});
module.exports = {
  createConversation,
  getUserConversations,
  getConversationById,
  renameConversation,
  deleteConversation,
};
