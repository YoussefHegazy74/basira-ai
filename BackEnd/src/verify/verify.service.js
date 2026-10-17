const AppError = require("../utils/AppError.js");
const HttpStatus = require("../utils/HttpStatusText.js");
const usageService = require("../usage/usage.service.js");

// Talk to model
const askModel = async (text, explanation = "") => {
  try {
    const response = await fetch(`${process.env.AI_URL}/api/verify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, explanation }),
      signal: AbortSignal.timeout(60000),
    });

    if (!response.ok) {
      throw new Error(`AI server responded with ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("AI model failed:", error.message);
    throw new AppError(502, "AI model failed", HttpStatus.ERROR);
  }
};

// Verify text
const verify = async (text, explanation = "", userId, guestKey) => {
  if (!text) {
    throw new AppError(400, "Text is required", HttpStatus.BAD_REQUEST);
  }

  if (!userId) {
    return await guestVerify(text, explanation, guestKey);
  }

  return await userVerify(text, explanation, userId, guestKey);
};

// Guest verification
const guestVerify = async (text, explanation, guestKey) => {
  const { ownerKey, limit } = usageService.identify(null, guestKey);

  await usageService.consume(ownerKey, limit);

  return await askModel(text, explanation);
};

// Authenticated user verification
const userVerify = async (text, explanation, userId, guestKey) => {
  const { ownerKey, limit } = usageService.identify(userId, null);
  await usageService.consume(ownerKey, limit);

  return await askModel(text, explanation);
};

module.exports = { verify };
