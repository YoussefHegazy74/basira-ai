// usage.routes.js
const express = require("express");
const authMiddleware = require("../middleware/auth.middleware.js");
const usageController = require("./usage.controller.js");
const restrictTo = require("../middleware/authorization.middleware.js");
const router = express.Router();

router.get("", authMiddleware.protect, usageController.getMyUsage);
router.post(
  "/reset/:userId",
  authMiddleware.protect,
  restrictTo("admin"),
  usageController.resetUsage,
);
router.post(
  "/reset-all",
  authMiddleware.protect,
  restrictTo("admin"),
  usageController.resetAllUsage,
);
module.exports = router;
