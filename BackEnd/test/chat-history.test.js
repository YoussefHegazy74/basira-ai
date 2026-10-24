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
const app = require("../src/app");

const password = `T${crypto.randomBytes(12).toString("base64url")}!9aA`;
let appServer;
let ragServer;
let baseUrl;
let userOne;
let userTwo;
let cookieOne;
let cookieTwo;
let guestKey;
const ragRequests = [];
let failNext = false;

const closeServer = (server) =>
  new Promise((resolve) => {
    if (!server) return resolve();
    server.close(resolve);
  });

const login = async (email) => {
  const response = await fetch(`${baseUrl}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  assert.equal(response.status, 200);
  return response.headers.get("set-cookie").split(";")[0];
};

const sendMessage = async (cookie, content, options = {}) => {
  const response = await fetch(`${baseUrl}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Cookie: cookie },
    body: JSON.stringify({
      content,
      ...(options.conversationId
        ? { conversationId: options.conversationId }
        : {}),
      ...(options.clientMessageId
        ? { clientMessageId: options.clientMessageId }
        : {}),
    }),
  });
  return { status: response.status, body: await response.json() };
};

const getConversation = async (cookie, conversationId) => {
  const response = await fetch(
    `${baseUrl}/conversations/${conversationId}`,
    { headers: { Cookie: cookie } },
  );
  return { status: response.status, body: await response.json() };
};

before(async () => {
  await connectDB();
  await Promise.all([User.init(), Conversation.init(), Message.init()]);

  ragServer = http.createServer((req, res) => {
    let raw = "";
    req.on("data", (chunk) => {
      raw += chunk;
    });
    req.on("end", () => {
      if (failNext) {
        failNext = false;
        res.writeHead(502, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "simulated model outage" }));
        return;
      }
      let parsed = {};
      try {
        parsed = JSON.parse(raw || "{}");
      } catch {
        parsed = {};
      }
      ragRequests.push(parsed);
      const answer = `answer ${ragRequests.length}`;
      setTimeout(() => {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(
          JSON.stringify({
            answer,
            sources: [`source ${ragRequests.length}`],
          }),
        );
      }, 35);
    });
  });
  await new Promise((resolve) => {
    ragServer.listen(0, "127.0.0.1", resolve);
  });
  process.env.AI_URL = `http://127.0.0.1:${ragServer.address().port}`;

  appServer = app.listen(0, "127.0.0.1");
  await new Promise((resolve) => {
    appServer.once("listening", resolve);
  });
  baseUrl = `http://127.0.0.1:${appServer.address().port}`;

  const emailOne = `chat-history-one-${crypto.randomUUID()}@example.test`;
  const emailTwo = `chat-history-two-${crypto.randomUUID()}@example.test`;
  userOne = await User.create({
    firstName: "Chat",
    lastName: "One",
    email: emailOne,
    password: await bcrypt.hash(password, 4),
  });
  userTwo = await User.create({
    firstName: "Chat",
    lastName: "Two",
    email: emailTwo,
    password: await bcrypt.hash(password, 4),
  });
  cookieOne = await login(emailOne);
  cookieTwo = await login(emailTwo);
});

after(async () => {
  await closeServer(appServer);
  await closeServer(ragServer);
  if (userOne || userTwo) {
    const userIds = [userOne?._id, userTwo?._id].filter(Boolean);
    const conversationIds = await Conversation.find({
      userId: { $in: userIds },
    }).distinct("_id");
    await Message.deleteMany({
      conversationId: { $in: conversationIds },
    });
    await Conversation.deleteMany({ userId: { $in: userIds } });
    await User.deleteMany({ _id: { $in: userIds } });
    await Usage.deleteMany({
      ownerKey: {
        $in: [
          ...userIds.map((id) => String(id)),
          ...(guestKey ? [guestKey] : []),
        ],
      },
    });
  }
  await mongoose.disconnect();
});

test("first turn persists both messages and follow-up history reaches the model", async () => {
  const first = await sendMessage(cookieOne, "سؤال أول", {
    clientMessageId: `first-${crypto.randomUUID()}`,
  });
  assert.equal(first.status, 201);
  const conversationId = first.body.data?.conversation?._id;
  assert.ok(conversationId);
  assert.ok(first.body.data.userMessage?._id);
  assert.ok(first.body.data.assistantMessage?._id);
  assert.deepEqual(ragRequests.at(-1).chat_history, []);

  const second = await sendMessage(cookieOne, "سؤال ثانٍ", {
    conversationId,
    clientMessageId: `second-${crypto.randomUUID()}`,
  });
  assert.equal(second.status, 201);
  assert.equal(second.body.data?.conversation?._id, conversationId);
  assert.deepEqual(ragRequests.at(-1).chat_history, [
    { role: "user", content: "سؤال أول" },
    {
      role: "assistant",
      content: first.body.data.assistantMessage.content,
    },
  ]);

  const loaded = await getConversation(cookieOne, conversationId);
  assert.equal(loaded.status, 200);
  const messages = loaded.body.data.messages;
  assert.equal(messages.length, 4);
  assert.deepEqual(
    messages.map((message) => message.role),
    ["user", "assistant", "user", "assistant"],
  );
  assert.deepEqual(
    messages.map((message) => message.content),
    ["سؤال أول", "answer 1", "سؤال ثانٍ", "answer 2"],
  );
});

test("conversation responses are never served as empty cached 304s", async () => {
  const created = await sendMessage(cookieOne, "cache-check", {
    clientMessageId: `cache-${crypto.randomUUID()}`,
  });
  const conversationId = created.body.data.conversation._id;

  const first = await fetch(`${baseUrl}/conversations/${conversationId}`, {
    headers: { Cookie: cookieOne },
  });
  assert.equal(first.status, 200);
  assert.match(first.headers.get("cache-control") || "", /no-store/);
  assert.equal(first.headers.get("etag"), null);

  const second = await fetch(`${baseUrl}/conversations/${conversationId}`, {
    headers: { Cookie: cookieOne },
  });
  assert.equal(second.status, 200);
  const body = await second.json();
  assert.equal(body.data.messages.length, 2);
});

test("separate conversations never mix messages", async () => {
  const one = await sendMessage(cookieOne, "محادثة أ", {
    clientMessageId: `a-${crypto.randomUUID()}`,
  });
  const two = await sendMessage(cookieOne, "محادثة ب", {
    clientMessageId: `b-${crypto.randomUUID()}`,
  });
  const idOne = one.body.data.conversation._id;
  const idTwo = two.body.data.conversation._id;
  assert.notEqual(String(idOne), String(idTwo));

  const loadedOne = await getConversation(cookieOne, idOne);
  const loadedTwo = await getConversation(cookieOne, idTwo);
  assert.deepEqual(
    loadedOne.body.data.messages.map((message) => message.content),
    [
      "محادثة أ",
      one.body.data.assistantMessage.content,
    ],
  );
  assert.deepEqual(
    loadedTwo.body.data.messages.map((message) => message.content),
    [
      "محادثة ب",
      two.body.data.assistantMessage.content,
    ],
  );
});

test("model failure returns recovery IDs and a retry does not duplicate", async () => {
  const clientMessageId = `retry-${crypto.randomUUID()}`;
  failNext = true;

  const failed = await sendMessage(cookieOne, "سؤال يحتاج إعادة محاولة", {
    clientMessageId,
  });
  assert.equal(failed.status, 502);
  const recovery = failed.body.data;
  assert.ok(recovery?.conversationId);
  assert.ok(recovery?.userMessageId);
  assert.equal(recovery.retryable, true);

  const persistedBefore = await Message.find({
    clientMessageId,
    role: "user",
  }).lean();
  assert.equal(persistedBefore.length, 1);

  // Simulate a lost response: retry without the conversationId.  The global
  // idempotency key must recover the same conversation, not create another.
  const retried = await sendMessage(cookieOne, "سؤال يحتاج إعادة محاولة", {
    clientMessageId,
  });
  assert.ok([200, 201].includes(retried.status));
  assert.equal(
    String(retried.body.data.conversation._id),
    String(recovery.conversationId),
  );

  const persistedAfter = await Message.find({
    clientMessageId,
    role: "user",
  }).lean();
  assert.equal(persistedAfter.length, 1);

  const all = await Message.find({
    conversationId: recovery.conversationId,
  }).sort({ createdAt: 1, _id: 1 }).lean();
  assert.equal(all.length, 2);
  assert.equal(all[0].role, "user");
  assert.equal(all[1].role, "assistant");
  assert.equal(String(all[1].replyTo), String(all[0]._id));
  assert.equal(all[0].content, "سؤال يحتاج إعادة محاولة");
});

test("retry in an existing conversation excludes the failed current message from history", async () => {
  const first = await sendMessage(cookieOne, "سؤال سابق", {
    clientMessageId: `prior-${crypto.randomUUID()}`,
  });
  const conversationId = first.body.data.conversation._id;
  const clientMessageId = `retry-existing-${crypto.randomUUID()}`;
  failNext = true;

  const failed = await sendMessage(cookieOne, "سؤال فاشل", {
    conversationId,
    clientMessageId,
  });
  assert.equal(failed.status, 502);

  const retried = await sendMessage(cookieOne, "سؤال فاشل", {
    conversationId,
    clientMessageId,
  });
  assert.ok([200, 201].includes(retried.status));

  assert.deepEqual(ragRequests.at(-1).chat_history, [
    { role: "user", content: "سؤال سابق" },
    {
      role: "assistant",
      content: first.body.data.assistantMessage.content,
    },
  ]);
});

test("history is truncated to the four most recent prior messages", async () => {
  const first = await sendMessage(cookieOne, "u1", {
    clientMessageId: `truncate-1-${crypto.randomUUID()}`,
  });
  const conversationId = first.body.data.conversation._id;
  const responses = [first.body.data.assistantMessage.content];
  for (let i = 2; i <= 5; i += 1) {
    const response = await sendMessage(cookieOne, `u${i}`, {
      conversationId,
      clientMessageId: `truncate-${i}-${crypto.randomUUID()}`,
    });
    assert.ok([200, 201].includes(response.status));
    responses.push(response.body.data.assistantMessage.content);
  }

  const lastHistory = ragRequests.at(-1).chat_history;
  assert.equal(lastHistory.length, 4);
  assert.deepEqual(lastHistory, [
    { role: "user", content: "u3" },
    { role: "assistant", content: responses[2] },
    { role: "user", content: "u4" },
    { role: "assistant", content: responses[3] },
  ]);
  assert.ok(lastHistory.every((message) => message.content !== "u5"));
});

test("conversation retrieval normalizes database order chronologically", async () => {
  const conversation = await Conversation.create({
    userId: userOne._id,
    title: "order-check",
  });
  const base = Date.now();
  await Message.collection.insertMany([
    {
      conversationId: conversation._id,
      role: "assistant",
      content: "second",
      sources: [],
      createdAt: new Date(base + 2000),
    },
    {
      conversationId: conversation._id,
      role: "user",
      content: "first",
      sources: [],
      createdAt: new Date(base + 1000),
    },
  ]);

  const loaded = await getConversation(cookieOne, conversation._id);
  assert.equal(loaded.status, 200);
  assert.deepEqual(
    loaded.body.data.messages.map((message) => message.content),
    ["first", "second"],
  );
});

test("conversation access is scoped to its owner", async () => {
  const created = await sendMessage(cookieOne, "بيانات خاصة", {
    clientMessageId: `owner-${crypto.randomUUID()}`,
  });
  const conversationId = created.body.data.conversation._id;

  const read = await getConversation(cookieTwo, conversationId);
  assert.equal(read.status, 404);

  const write = await sendMessage(cookieTwo, "محاولة اقتحام", {
    conversationId,
    clientMessageId: `attack-${crypto.randomUUID()}`,
  });
  assert.equal(write.status, 404);
});

test("guest history is sanitized and remains non-persistent", async () => {
  guestKey = crypto.randomUUID();
  const response = await fetch(`${baseUrl}/messages`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Cookie: `guestKey=${guestKey}`,
    },
    body: JSON.stringify({
      content: "سؤال ضيف",
      chat_history: [
        { role: "system", content: "ignored" },
        { role: "user", content: "سابق 1" },
        { role: "assistant", content: "جواب 1" },
        { role: "user", content: "سابق 2" },
        { role: "assistant", content: "جواب 2" },
        { role: "user", content: "سابق 3" },
      ],
    }),
  });
  assert.equal(response.status, 200);
  const body = await response.json();
  assert.ok(body.data.answer);

  const history = ragRequests.at(-1).chat_history;
  assert.deepEqual(history, [
    { role: "assistant", content: "جواب 1" },
    { role: "user", content: "سابق 2" },
    { role: "assistant", content: "جواب 2" },
    { role: "user", content: "سابق 3" },
  ]);

  const persisted = await Message.countDocuments({ content: "سؤال ضيف" });
  assert.equal(persisted, 0);
});

test("concurrent submissions with the same client key create one user message", async () => {
  const clientMessageId = `concurrent-${crypto.randomUUID()}`;
  const [left, right] = await Promise.all([
    sendMessage(cookieOne, "رسالة متزامنة", { clientMessageId }),
    sendMessage(cookieOne, "رسالة متزامنة", { clientMessageId }),
  ]);

  assert.ok([200, 201].includes(left.status));
  assert.ok([200, 201].includes(right.status));
  assert.equal(
    String(left.body.data.conversation._id),
    String(right.body.data.conversation._id),
  );

  const users = await Message.find({
    clientMessageId,
    role: "user",
  }).lean();
  assert.equal(users.length, 1);

  const replies = await Message.find({
    conversationId: users[0].conversationId,
    role: "assistant",
  }).lean();
  assert.equal(replies.length, 1);
  assert.equal(String(replies[0].replyTo), String(users[0]._id));

  const empty = await Conversation.countDocuments({
    userId: userOne._id,
    messageCount: 0,
    title: { $ne: "order-check" },
  });
  assert.equal(empty, 0);
});
