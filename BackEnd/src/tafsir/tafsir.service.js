const AppError = require("../utils/AppError.js");
const HttpStatus = require("../utils/HttpStatusText.js");
const usageService = require("../usage/usage.service.js");

const RAG_TIMEOUT_MS = 60000;
const UNAVAILABLE_MESSAGE =
  "Tafsir service is unavailable. Please try again later.";
const TIMEOUT_MESSAGE = "Tafsir service timed out. Please try again.";
const INVALID_RESPONSE_MESSAGE = "Tafsir service returned an invalid response.";

///talk to model
const askModel = async (ayah) => {
  let response;

  try {
    response = await fetch(`${process.env.AI_URL}/api/tafsir`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ayah }),
      signal: AbortSignal.timeout(RAG_TIMEOUT_MS),
    });
  } catch (error) {
    const timedOut =
      error.name === "TimeoutError" || error.name === "AbortError";
    console.error("Tafsir upstream request failed:", error.message);
    throw new AppError(
      timedOut ? 504 : 502,
      timedOut ? TIMEOUT_MESSAGE : UNAVAILABLE_MESSAGE,
      HttpStatus.ERROR,
    );
  }

  if (!response.ok) {
    console.error(`Tafsir upstream responded with HTTP ${response.status}`);
    throw new AppError(502, UNAVAILABLE_MESSAGE, HttpStatus.ERROR);
  }

  try {
    return await response.json();
  } catch (error) {
    console.error("Tafsir upstream returned invalid JSON:", error.message);
    throw new AppError(502, INVALID_RESPONSE_MESSAGE, HttpStatus.ERROR);
  }
};

/// get tafsir
const tafsir = async (ayah, userId, guestKey) => {
  if (!ayah) {
    throw new AppError(400, "Ayah is required", HttpStatus.BAD_REQUEST);
  }
  if (!userId) {
    return await gusetTafsir(ayah, userId, guestKey);
  }
  return await userTafsir(ayah, userId, guestKey);
};
/// guesttafsir
const gusetTafsir = async (ayah, userId, guestKey) => {
  const { ownerKey } = usageService.identify(null, guestKey);
  await usageService.consume(ownerKey);
  return await askModel(ayah);
};
/// usertafsir
const userTafsir = async (ayah, userId, guestKey) => {
  const { ownerKey } = usageService.identify(userId, null);
  await usageService.consume(ownerKey);
  return await askModel(ayah);
};

module.exports = { tafsir, askModel };
