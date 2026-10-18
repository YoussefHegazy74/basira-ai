const express = require("express");
const messageController = require("./message.controller.js");
const authMiddleware = require("../middleware/auth.middleware.js");
const router = express.Router();

router.post("/", authMiddleware.optionalAuth, messageController.sendMessage);
router.delete("/:id", authMiddleware.protect, messageController.deleteMessage);

module.exports = router;
