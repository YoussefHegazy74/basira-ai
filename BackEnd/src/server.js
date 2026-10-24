const dotenv = require("dotenv");
const path = require("path");

dotenv.config({
  path: path.resolve(__dirname, "../../.env"),
});

const mongoose = require("mongoose");
const app = require("./app");
const connectDB = require("./config/dbConnect");
const User = require("./user/user.model");
const Conversation = require("./conversation/conversation.model");
const Message = require("./message/message.model");

const PORT = process.env.PORT || 3000;
const REQUIRED_ENV = ["DB_URI", "DB_NAME", "JWT_SECRET", "AI_URL"];

const validateConfiguration = () => {
  const missing = REQUIRED_ENV.filter((name) => !process.env[name]);
  if (missing.length) {
    throw new Error(`Missing required environment: ${missing.join(", ")}`);
  }
  if (
    process.env.NODE_ENV === "production" &&
    String(process.env.JWT_SECRET).length < 32
  ) {
    throw new Error("JWT_SECRET must be at least 32 characters in production");
  }
};

const startServer = async () => {
  try {
    validateConfiguration();
    await connectDB();
    // Unique idempotency indexes must exist before accepting traffic.  A
    // missing index would allow concurrent retries to persist duplicates.
    await Promise.all([User.init(), Conversation.init(), Message.init()]);

    const server = app.listen(PORT, () => {
      console.log(`Server is running on port ${PORT}`);
    });

    let shuttingDown = false;
    const shutdown = async (signal) => {
      if (shuttingDown) return;
      shuttingDown = true;
      console.log(`Received ${signal}; shutting down`);
      server.close(async () => {
        try {
          await mongoose.disconnect();
        } finally {
          process.exit(0);
        }
      });
      setTimeout(() => process.exit(1), 10_000).unref();
    };
    process.once("SIGTERM", () => shutdown("SIGTERM"));
    process.once("SIGINT", () => shutdown("SIGINT"));
  } catch (error) {
    console.error("Failed to start server:", error.message);
    process.exit(1);
  }
};

startServer();
