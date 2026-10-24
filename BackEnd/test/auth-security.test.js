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
const Usage = require("../src/usage/usage.model");
const app = require("../src/app");

const password = `T${crypto.randomBytes(12).toString("base64url")}!9aA`;
const TEST_DB_NAME = `basira_authsec_${crypto.randomUUID()}`;
let server;
let baseUrl;
let admin;
let target;
let adminCookie;

const login = async (email) => {
  const response = await fetch(`${baseUrl}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  return {
    status: response.status,
    body: await response.json(),
    cookie: response.headers.get("set-cookie")?.split(";")[0],
  };
};

before(async () => {
  // Isolated test database: this suite must be able to test last-admin
  // behavior without counting or modifying real administrators.
  process.env.DB_NAME = TEST_DB_NAME;
  await connectDB();
  await User.init();
  server = app.listen(0, "127.0.0.1");
  await new Promise((resolve) => server.once("listening", resolve));
  baseUrl = `http://127.0.0.1:${server.address().port}`;

  const adminEmail = `authsec-admin-${crypto.randomUUID()}@example.test`;
  const targetEmail = `authsec-target-${crypto.randomUUID()}@example.test`;
  admin = await User.create({
    firstName: "Auth",
    lastName: "Admin",
    email: adminEmail,
    password: await bcrypt.hash(password, 4),
    role: "admin",
  });
  target = await User.create({
    firstName: "Auth",
    lastName: "Target",
    email: targetEmail,
    password: await bcrypt.hash(password, 4),
  });
  const loggedIn = await login(adminEmail);
  assert.equal(loggedIn.status, 200);
  adminCookie = loggedIn.cookie;
});

after(async () => {
  if (server) await new Promise((resolve) => server.close(resolve));
  const ids = [admin?._id, target?._id].filter(Boolean);
  await User.deleteMany({ _id: { $in: ids } });
  await Usage.deleteMany({ ownerKey: { $in: ids.map(String) } });
  await mongoose.connection.dropDatabase();
  await mongoose.disconnect();
});

test("deactivation revokes the session and cannot be undone by login", async () => {
  const targetLogin = await login(target.email);
  assert.equal(targetLogin.status, 200);

  const deactivate = await fetch(`${baseUrl}/auth/deactivate-account`, {
    method: "PATCH",
    headers: { Cookie: targetLogin.cookie },
  });
  assert.equal(deactivate.status, 200);
  assert.match(deactivate.headers.get("set-cookie") || "", /token=;/);

  const stored = await User.findById(target._id);
  assert.equal(stored.isActive, false);
  assert.ok(stored.expiresAt instanceof Date);

  const reused = await fetch(`${baseUrl}/usage`, {
    headers: { Cookie: targetLogin.cookie },
  });
  assert.equal(reused.status, 401);
  assert.equal((await reused.json()).message, "Account is deactivated");

  const relogin = await login(target.email);
  assert.equal(relogin.status, 403);
  assert.equal(relogin.body.message, "Account is deactivated");
  const stillInactive = await User.findById(target._id);
  assert.equal(stillInactive.isActive, false);
});

test("the last administrator cannot deactivate their own account", async () => {
  const response = await fetch(`${baseUrl}/auth/deactivate-account`, {
    method: "PATCH",
    headers: { Cookie: adminCookie },
  });
  assert.equal(response.status, 400);
  assert.equal(
    (await response.json()).message,
    "Cannot deactivate the last administrator",
  );
  const stored = await User.findById(admin._id);
  assert.equal(stored.isActive, true);
});

test("the last administrator cannot be deactivated through user update", async () => {
  const response = await fetch(`${baseUrl}/users/${admin._id}`, {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
      Cookie: adminCookie,
    },
    body: JSON.stringify({ isActive: false }),
  });
  assert.equal(response.status, 400);
  assert.equal(
    (await response.json()).message,
    "Cannot deactivate the last administrator",
  );
});

test("an administrator cannot delete their own account", async () => {
  const response = await fetch(`${baseUrl}/users/${admin._id}`, {
    method: "DELETE",
    headers: { Cookie: adminCookie },
  });
  assert.equal(response.status, 400);
  assert.equal(
    (await response.json()).message,
    "You cannot delete your own account",
  );
});

test("only an admin can reactivate an account, and reactivation clears TTL", async () => {
  const reactivate = await fetch(`${baseUrl}/users/${target._id}`, {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
      Cookie: adminCookie,
    },
    body: JSON.stringify({ isActive: true }),
  });
  assert.equal(reactivate.status, 200);

  const stored = await User.findById(target._id);
  assert.equal(stored.isActive, true);
  assert.equal(stored.expiresAt, null);

  const relogin = await login(target.email);
  assert.equal(relogin.status, 200);
});
