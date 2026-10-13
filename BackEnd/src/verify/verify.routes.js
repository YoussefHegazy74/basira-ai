const express = require("express");
const router = express.Router();
const verifyController = require("./verify.controller.js");
const authMiddleware = require("../middleware/auth.middleware.js");

router.post("/", authMiddleware.optionalAuth, verifyController.getVerify);

module.exports = router;
