const { test, before, after } = require("node:test");
const assert = require("node:assert/strict");
const path = require("path");
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
const app = require("../src/app");

const password = `T${crypto.randomBytes(12).toString("base64url")}!9aA`;
let server;
let baseUrl;
let userOne;
let userTwo;
let cookieOne;

const login = async (email) => {
  const response = await fetch(`${baseUrl}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  assert.equal(response.status, 200);
  return response.headers.get("set-cookie").split(";")[0];
};

const patchUser = async (cookie, id, body) => {
  const response = await fetch(`${baseUrl}/users/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json", Cookie: cookie },
    body: JSON.stringify(body),
  });
  return { status: response.status, body: await response.json() };
};

before(async () => {
  await connectDB();
  await User.init();

  server = app.listen(0, "127.0.0.1");
  await new Promise((resolve) => server.once("listening", resolve));
  baseUrl = `http://127.0.0.1:${server.address().port}`;

  const emailOne = `authz-one-${crypto.randomUUID()}@example.test`;
  const emailTwo = `authz-two-${crypto.randomUUID()}@example.test`;
  userOne = await User.create({
    firstName: "Authz",
    lastName: "One",
    email: emailOne,
    password: await bcrypt.hash(password, 4),
  });
  userTwo = await User.create({
    firstName: "Authz",
    lastName: "Two",
    email: emailTwo,
    password: await bcrypt.hash(password, 4),
  });
  cookieOne = await login(emailOne);
});

after(async () => {
  if (server) await new Promise((resolve) => server.close(resolve));
  await User.deleteMany({
    _id: { $in: [userOne?._id, userTwo?._id].filter(Boolean) },
  });
  await mongoose.disconnect();
});

test("a user cannot update another user", async () => {
  const result = await patchUser(cookieOne, userTwo._id, {
    firstName: "Hacked",
  });
  assert.equal(result.status, 403);
  const stored = await User.findById(userTwo._id);
  assert.equal(stored.firstName, "Authz");
});

test("self-update cannot escalate role or change password", async () => {
  const updated = await patchUser(cookieOne, userOne._id, {
    firstName: "Updated",
    role: "admin",
    password: "attacker-controlled",
  });
  assert.equal(updated.status, 200);
  assert.equal(updated.body.data.firstName, "Updated");
  assert.equal(updated.body.data.role, "user");

  const stored = await User.findById(userOne._id).select("+password");
  assert.equal(stored.role, "user");
  assert.ok(await bcrypt.compare(password, stored.password));
});
