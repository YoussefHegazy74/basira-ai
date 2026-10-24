const { test, before, after } = require("node:test");
const assert = require("node:assert/strict");
const path = require("path");
const http = require("node:http");
const crypto = require("crypto");
const bcrypt = require("bcrypt");
const dotenv = require("dotenv");

dotenv.config({
  path: path.resolve(__dirname, "../../.env"),
  quiet: true,
});

const mongoose = require("mongoose");
const connectDB = require("../src/config/dbConnect");
const User = require("../src/user/user.model");
const Conversation = require("../src/conversation/conversation.model");
const Message = require("../src/message/message.model");
const Usage = require("../src/usage/usage.model");
const usageService = require("../src/usage/usage.service");
const app = require("../src/app");

const FORMER_GUEST_LIMIT = 5;
const FORMER_USER_LIMIT = 10;
const password = `T${crypto.randomBytes(12).toString("base64url")}!9aA`;

let appServer;
let ragServer;
let baseUrl;
let email;
let userId;
let userCookie;
let guestKey;
const createdOwnerKeys = [];

const startServer = async (handler) => {
  const server = http.createServer(handler);
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  return server;
};

const closeServer = (server) =>
  new Promise((resolve) => {
    if (!server) return resolve();
    server.close(resolve);
  });

const seedUsage = async (ownerKey, count) => {
  createdOwnerKeys.push(ownerKey);
  await Usage.findOneAndUpdate(
    { ownerKey },
    { $set: { count, expiresAt: new Date(Date.now() + 60 * 60 * 1000) } },
    { upsert: true, runValidators: true },
  );
};

const messageRequest = (content, options = {}) =>
  fetch(`${baseUrl}/messages`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(options.cookie ? { Cookie: options.cookie } : {}),
    },
    body: JSON.stringify({
      content,
      ...(options.conversationId
        ? { conversationId: options.conversationId }
        : {}),
    }),
  });

before(async () => {
  await connectDB();

  ragServer = await startServer((req, res) => {
    req.resume();
    req.on("end", () => {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(
        JSON.stringify({
          answer: "إجابة تجريبية من الخدمة الوهمية",
          sources: ["مصدر تجريبي"],
        }),
      );
    });
  });
  process.env.AI_URL = `http://127.0.0.1:${ragServer.address().port}`;

  email = `usage-regression-${crypto.randomUUID()}@example.test`;
  const user = await User.create({
    firstName: "Usage",
    lastName: "Regression",
    email,
    password: await bcrypt.hash(password, 4),
  });
  userId = user._id;

  appServer = app.listen(0, "127.0.0.1");
  await new Promise((resolve, reject) => {
    appServer.once("listening", resolve);
    appServer.once("error", reject);
  });
  baseUrl = `http://127.0.0.1:${appServer.address().port}`;

  const loginResponse = await fetch(`${baseUrl}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  assert.equal(loginResponse.status, 200);
  userCookie = loginResponse.headers.get("set-cookie").split(";")[0];
  assert.match(userCookie, /^token=/);
});

after(async () => {
  if (appServer) {
    await closeServer(appServer);
  }
  if (userId) {
    await Message.deleteMany({ conversationId: { $in: await Conversation.find({ userId }).distinct("_id") } });
    await Conversation.deleteMany({ userId });
    await User.deleteOne({ _id: userId });
  }
  await Usage.deleteMany({ ownerKey: { $in: createdOwnerKeys } });
  await closeServer(ragServer);
  await mongoose.disconnect();
});

test("authenticated chat exceeds the former daily quota and persists messages", async () => {
  createdOwnerKeys.push(String(userId));
  await seedUsage(String(userId), FORMER_USER_LIMIT);

  const first = await messageRequest("سؤال تجريبي أول", {
    cookie: userCookie,
  });
  assert.equal(first.status, 201);
  const firstBody = await first.json();
  const conversationId = firstBody.data?.conversation?._id;
  assert.ok(conversationId);

  const second = await messageRequest("سؤال تجريبي ثانٍ", {
    cookie: userCookie,
    conversationId,
  });
  assert.equal(second.status, 201);

  const third = await messageRequest("سؤال تجريبي ثالث", {
    cookie: userCookie,
    conversationId,
  });
  assert.equal(third.status, 201);

  const persisted = await fetch(`${baseUrl}/conversations/${conversationId}`, {
    headers: { Cookie: userCookie },
  });
  assert.equal(persisted.status, 200);
  const persistedBody = await persisted.json();
  const messages = persistedBody.data.messages;
  assert.equal(messages.length, 6);
  assert.deepEqual(
    messages.filter((message) => message.role === "user").map((m) => m.content),
    ["سؤال تجريبي أول", "سؤال تجريبي ثانٍ", "سؤال تجريبي ثالث"],
  );
  assert.equal(
    messages.filter((message) => message.role === "assistant").length,
    3,
  );

  const usageResponse = await fetch(`${baseUrl}/usage`, {
    headers: { Cookie: userCookie },
  });
  assert.equal(usageResponse.status, 200);
  const usageBody = await usageResponse.json();
  assert.equal(usageBody.data.used, FORMER_USER_LIMIT + 3);
  assert.equal(usageBody.data.limit, undefined);
  assert.equal(usageBody.data.remaining, undefined);

  const storedUsage = await Usage.findOne({ ownerKey: String(userId) }).lean();
  assert.equal(storedUsage.count, FORMER_USER_LIMIT + 3);
});

test("guest chat exceeds the former daily quota", async () => {
  guestKey = crypto.randomUUID();
  await seedUsage(guestKey, FORMER_GUEST_LIMIT);

  for (let index = 0; index < 3; index += 1) {
    const response = await messageRequest(`رسالة ضيف ${index + 1}`, {
      cookie: `guestKey=${guestKey}`,
    });
    assert.equal(response.status, 200);
    const body = await response.json();
    assert.equal(body.data.answer, "إجابة تجريبية من الخدمة الوهمية");
  }

  const storedUsage = await Usage.findOne({ ownerKey: guestKey }).lean();
  assert.equal(storedUsage.count, FORMER_GUEST_LIMIT + 3);
});

test("usage service records beyond the former limits without throwing", async () => {
  const ownerKey = `usage-service-${crypto.randomUUID()}`;
  await seedUsage(ownerKey, FORMER_USER_LIMIT);

  await usageService.consume(ownerKey);
  await usageService.consume(ownerKey);

  const storedUsage = await Usage.findOne({ ownerKey }).lean();
  assert.equal(storedUsage.count, FORMER_USER_LIMIT + 2);
});

test("expired usage records reset without blocking the next request", async () => {
  const ownerKey = `usage-expired-${crypto.randomUUID()}`;
  createdOwnerKeys.push(ownerKey);
  await Usage.create({
    ownerKey,
    count: 99,
    expiresAt: new Date(Date.now() - 1000),
  });

  await usageService.consume(ownerKey);

  const storedUsage = await Usage.findOne({ ownerKey }).lean();
  assert.equal(storedUsage.count, 1);
});

test("authentication and input validation still reject invalid requests", async () => {
  const unauthenticated = await fetch(`${baseUrl}/conversations`);
  assert.equal(unauthenticated.status, 401);

  const invalidToken = await messageRequest("رسالة", {
    cookie: "token=not-a-real-token",
  });
  assert.equal(invalidToken.status, 401);
  assert.equal((await invalidToken.json()).message, "Invalid token");

  const tooLong = await messageRequest("ا".repeat(501), {
    cookie: userCookie,
  });
  assert.equal(tooLong.status, 400);
  assert.match((await tooLong.json()).message, /maximum length/i);
});
