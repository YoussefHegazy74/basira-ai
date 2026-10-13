const Usage = require("./usage.model");
const AppError = require("../utils/appError.js");
const HttpStatus = require("../utils/HttpStatusText.js");

const Guest_Usage_limit = 5;
const User_Usage_limit = 10;

////////// identify consumer
const identify = (UserId, guestKey) => {
  return UserId
    ? { ownerKey: UserId, limit: User_Usage_limit }
    : { ownerKey: guestKey, limit: Guest_Usage_limit };
};
////////// consume one usage
const consume = async (ownerKey, limit) => {
  const increment = () =>
    Usage.findOneAndUpdate(
      { ownerKey, expiresAt: { $gt: new Date() }, count: { $lt: limit } },
      { $inc: { count: 1 } },
    );
  if (await increment()) return;

  await Usage.deleteOne({ ownerKey, expiresAt: { $lte: new Date() } });

  try {
    await Usage.create({ ownerKey, count: 1 });
    return;
  } catch (err) {
    if (err.code !== 11000) throw err;
  }

  if (await increment()) return;

  throw new AppError(429, "Daily usage limit reached", httpStatusText.FAIL);
};
////////// get usage for a user
const getUsage = async (userId) => {
  const usage = await Usage.findOne({
    ownerKey: String(userId),
    expiresAt: { $gt: new Date() },
  }).lean();

  const used = usage ? usage.count : 0;

  return {
    limit: User_Usage_limit,
    used,
    remaining: Math.max(User_Usage_limit - used, 0),
    resetsAt: usage ? usage.expiresAt : null,
  };
};

///
module.exports = {
  identify,
  consume,
  getUsage,
};
