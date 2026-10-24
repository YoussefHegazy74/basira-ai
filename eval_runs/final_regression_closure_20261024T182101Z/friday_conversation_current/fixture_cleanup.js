const fs = require("fs");
const dotenv = require("dotenv");

dotenv.config({ path: "/run/media/joe/Data/basira-ai/.env", quiet: true });

const mongoose = require("mongoose");
const connectDB = require("/run/media/joe/Data/basira-ai/BackEnd/src/config/dbConnect");
const User = require("/run/media/joe/Data/basira-ai/BackEnd/src/user/user.model");
const Conversation = require("/run/media/joe/Data/basira-ai/BackEnd/src/conversation/conversation.model");
const Message = require("/run/media/joe/Data/basira-ai/BackEnd/src/message/message.model");
const Usage = require("/run/media/joe/Data/basira-ai/BackEnd/src/usage/usage.model");

(async () => {
  const fixture = JSON.parse(
    fs.readFileSync("/tmp/opencode/friday_fixture.json", "utf8"),
  );
  await connectDB();
  const ids = await Conversation.find({ userId: fixture.userId }).distinct("_id");
  const messages = await Message.deleteMany({ conversationId: { $in: ids } });
  const conversations = await Conversation.deleteMany({ userId: fixture.userId });
  const user = await User.deleteOne({ _id: fixture.userId });
  const usage = await Usage.deleteMany({ ownerKey: String(fixture.userId) });
  console.log(
    JSON.stringify({
      conversations: conversations.deletedCount,
      messages: messages.deletedCount,
      user: user.deletedCount,
      usage: usage.deletedCount,
    }),
  );
  await mongoose.disconnect();
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
