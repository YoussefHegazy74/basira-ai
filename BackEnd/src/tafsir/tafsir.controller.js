const HttpStatus = require("../utils/HttpStatusText.js");
const asyncWrapper = require("../middleware/Async_wrapper.js");
const tafsirService = require("./tafsir.service.js");
const crypto = require("node:crypto");
const getTafsir = asyncWrapper(async (req, res, next) => {
  let guestKey;
  if (!req.user) {
    guestKey = req.cookies.guestKey;

    if (!guestKey) {
      guestKey = crypto.randomUUID();
      res.cookie("guestKey", guestKey, {
        httpOnly: true,
        sameSite: "lax",
        maxAge: 24 * 60 * 60 * 1000,
      });
    }
  }
  const result = await tafsirService.tafsir(
    req.body.ayah,
    req.user?._id,
    guestKey,
  );
  res.status(req.user ? 201 : 200).json({
    status: HttpStatus.SUCCESS,
    data: result,
  });
});
module.exports = { getTafsir };
