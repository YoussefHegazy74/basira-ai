const messageService = require("./message.service");
const HttpStatus = require("../utils/HttpStatusText.js");
const asyncWrapper = require("../middleware/Async_wrapper.js");
const crypto = require("node:crypto");

// send message (user or guest)
const sendMessage = asyncWrapper(async (req, res, next) => {
  let guestKey;
  if (!req.user) {
    guestKey = req.cookies.guestKey;

    if (!guestKey) {
      guestKey = crypto.randomUUID();
      res.cookie("guestKey", guestKey, {
        httpOnly: true,
        sameSite: "lax",
        maxAge: 24 * 60 * 60 * 1000,
      });
    }
  }
  const result = await messageService.sendMessage({
    userId: req.user?._id,
    conversationId: req.body.conversationId,
    content: req.body.content,
    chatHistory: req.body.chat_history ?? req.body.chatHistory,
    clientMessageId: req.body.clientMessageId,
    guestKey,
  });

  res.status(req.user ? (result.replayed ? 200 : 201) : 200).json({
    status: HttpStatus.SUCCESS,
    data: result,
  });
});

// delete message
const deleteMessage = asyncWrapper(async (req, res, next) => {
  await messageService.deleteMessage({
    messageId: req.params.id,
    userId: req.user._id,
  });

  res.status(204).json({
    status: HttpStatus.SUCCESS,
    data: null,
  });
});

module.exports = { sendMessage, deleteMessage };
