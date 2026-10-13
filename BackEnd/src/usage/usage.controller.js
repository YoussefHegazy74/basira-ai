// usage.controller.js
const asyncWrapper = require("../middleware/Async_wrapper");
const httpStatusText = require("../utils/HttpStatusText");
const usageService = require("./usage.service");

const getMyUsage = asyncWrapper(async (req, res) => {
  const data = await usageService.getUsage(req.user._id);
  res.status(200).json({ status: httpStatusText.SUCCESS, data });
});
//// reset usage for a user
const resetUsage = asyncWrapper(async (req, res) => {
  await usageService.resetUsage(req.params.userId);
  res.status(204).json({ status: httpStatusText.SUCCESS, data: null });
});
// reset usage for all users
const resetAllUsage = asyncWrapper(async (req, res) => {
  await usageService.resetAllUsage();
  res.status(204).json({ status: httpStatusText.SUCCESS, data: null });
});

module.exports = { getMyUsage, resetUsage, resetAllUsage };
