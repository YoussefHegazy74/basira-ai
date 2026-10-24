const Usage = require("./usage.model");

////////// identify consumer
const identify = (userId, guestKey) => {
  return { ownerKey: String(userId || guestKey) };
};

////////// record one usage without enforcing a daily cap
const consume = async (ownerKey) => {
  const increment = () =>
    Usage.findOneAndUpdate(
      { ownerKey, expiresAt: { $gt: new Date() } },
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

  await Usage.updateOne({ ownerKey }, { $inc: { count: 1 } });
};

////////// get usage for a user
const getUsage = async (userId) => {
  const usage = await Usage.findOne({
    ownerKey: String(userId),
    expiresAt: { $gt: new Date() },
  }).lean();

  return {
    used: usage ? usage.count : 0,
    resetsAt: usage ? usage.expiresAt : null,
  };
};

/// reset usage for a user
const resetUsage = async (userId) => {
  await Usage.deleteOne({ ownerKey: String(userId) });
};
// reset usage for all users
const resetAllUsage = async () => {
  await Usage.deleteMany({});
};
module.exports = {
  identify,
  consume,
  getUsage,
  resetUsage,
  resetAllUsage,
};
