const crypto = require("crypto");
const bcrypt = require("bcrypt");
const dotenv = require("dotenv");
const fs = require("fs");

dotenv.config({ path: "/run/media/joe/Data/basira-ai/.env", quiet: true });

const mongoose = require("mongoose");
const connectDB = require("/run/media/joe/Data/basira-ai/BackEnd/src/config/dbConnect");
const User = require("/run/media/joe/Data/basira-ai/BackEnd/src/user/user.model");

(async () => {
  await connectDB();
  const email = `friday-e2e-${crypto.randomUUID()}@example.test`;
  const password = `T${crypto.randomBytes(10).toString("base64url")}!9aA`;
  const user = await User.create({
    firstName: "Friday",
    lastName: "E2E",
    email,
    password: await bcrypt.hash(password, 4),
  });
  fs.writeFileSync(
    "/tmp/opencode/friday_fixture.json",
    JSON.stringify({ userId: String(user._id), email, password }, null, 2),
  );
  console.log("fixture user", String(user._id), email);
  await mongoose.disconnect();
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
