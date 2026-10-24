const AppError = require("../utils/AppError.js");
const HttpStatus = require("../utils/HttpStatusText.js");
const usageService = require("../usage/usage.service.js");
///talk to model
const askModel = async (ayah) => {
  try {
    const response = await fetch(`${process.env.AI_URL}/api/tafsir`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ayah }),
      signal: AbortSignal.timeout(60000),
    });

    if (!response.ok) {
      throw new Error(`استجاب خادم الذكاء الاصطناعي بالرمز ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error("AI model failed:", error.message);
    throw new AppError(502, "فشل نموذج الذكاء الاصطناعي", HttpStatus.ERROR);
  }
};

/// get tafsir
const tafsir = async (ayah, userId, guestKey) => {
  if (!ayah) {
    throw new AppError(400, "الآية مطلوبة", HttpStatus.BAD_REQUEST);
  }
  if (!userId) {
    return await gusetTafsir(ayah, userId, guestKey);
  }
  return await userTafsir(ayah, userId, guestKey);
};
/// guesttafsir
const gusetTafsir = async (ayah, userId, guestKey) => {
  const { ownerKey, limit } = usageService.identify(null, guestKey);
  await usageService.consume(ownerKey, limit);
  return await askModel(ayah);
};
/// usertafsir
const userTafsir = async (ayah, userId, guestKey) => {
  const { ownerKey, limit } = usageService.identify(userId, null);
  await usageService.consume(ownerKey, limit);
  return await askModel(ayah);
};

module.exports = { tafsir };
