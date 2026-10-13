// usage.routes.js
const express = require("express");
const authMiddleware = require("../middleware/auth.middleware.js");
const { getMyUsage } = require("./usage.controller");

const router = express.Router();

router.get("/", authMiddleware.protect, getMyUsage);

module.exports = router;
