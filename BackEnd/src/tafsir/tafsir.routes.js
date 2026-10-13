const express = require("express");
const router = express.Router();
const tafsirController = require("./tafsir.controller.js");
const authMiddleware = require("../middleware/auth.middleware.js");

router.post("/", authMiddleware.optionalAuth, tafsirController.getTafsir);

module.exports = router;
