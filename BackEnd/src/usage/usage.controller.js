// usage.controller.js
const asyncWrapper = require("../middleware/Async_wrapper");
const httpStatusText = require("../utils/HttpStatusText");
const usageService = require("./usage.service");

const getMyUsage = asyncWrapper(async (req, res) => {
  const data = await usageService.getUsage(req.user._id);
  res.status(200).json({ status: httpStatusText.SUCCESS, data });
});

module.exports = { getMyUsage };
