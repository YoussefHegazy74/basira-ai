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
const origin = "http://localhost:5173";

let server;
let baseUrl;
let userId;
let email;

before(async () => {
  await connectDB();

  email = `login-regression-${crypto.randomUUID()}@example.test`;
  const user = await User.create({
    firstName: "Login",
    lastName: "Regression",
    email,
    password: await bcrypt.hash(password, 4),
    role: "admin",
  });
  userId = user._id;

  server = app.listen(0, "127.0.0.1");
  await new Promise((resolve, reject) => {
    server.once("listening", resolve);
    server.once("error", reject);
  });
  baseUrl = `http://127.0.0.1:${server.address().port}`;
});

after(async () => {
  if (server) {
    await new Promise((resolve) => server.close(resolve));
  }
  if (userId) {
    await User.deleteOne({ _id: userId });
  }
  await mongoose.disconnect();
});

test("health and readiness endpoints report the running process", async () => {
  const health = await fetch(`${baseUrl}/health`);
  assert.equal(health.status, 200);
  assert.deepEqual(await health.json(), { status: "ok" });

  const ready = await fetch(`${baseUrl}/ready`);
  assert.equal(ready.status, 200);
  assert.deepEqual(await ready.json(), { status: "ready" });
});

test("user lookup and login succeed against the configured database", async () => {
  const response = await fetch(`${baseUrl}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Origin: origin },
    body: JSON.stringify({ email, password }),
  });

  assert.equal(response.status, 200);
  assert.equal(
    response.headers.get("access-control-allow-origin"),
    origin,
  );

  const body = await response.json();
  assert.equal(body.status, "Success");
  assert.equal(body.user.email, email);
  assert.equal(body.user.id, String(userId));
  assert.equal(body.user.password, undefined);

  const setCookie = response.headers.get("set-cookie");
  assert.match(setCookie, /token=/);
  assert.match(setCookie, /HttpOnly/i);
  const cookie = setCookie.split(";")[0];

  const protectedResponse = await fetch(`${baseUrl}/users`, {
    headers: { Cookie: cookie, Origin: origin },
  });
  assert.equal(protectedResponse.status, 200);

  const protectedBody = await protectedResponse.json();
  assert.equal(protectedBody.status, "Success");
  assert.ok(protectedBody.data.some((user) => user.email === email));
});

test("CORS does not reflect an unknown origin", async () => {
  const response = await fetch(`${baseUrl}/auth/login`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Origin: "https://evil.example",
    },
    body: JSON.stringify({ email, password }),
  });
  assert.equal(response.headers.get("access-control-allow-origin"), null);
});

test("forgot-password does not reveal an unknown account", async () => {
  const response = await fetch(`${baseUrl}/auth/forgotpassword`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      email: `missing-${crypto.randomUUID()}@example.test`,
    }),
  });
  assert.equal(response.status, 200);
  const body = await response.json();
  assert.equal(
    body.message,
    "إذا كان البريد الإلكتروني موجودًا، فقد تم إرسال رابط إعادة تعيين كلمة المرور.",
  );
});

test("login rejects an incorrect password", async () => {
  const response = await fetch(`${baseUrl}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password: `${password}-wrong` }),
  });

  assert.equal(response.status, 401);
  assert.equal((await response.json()).message, "البريد الإلكتروني أو كلمة المرور غير صحيحة");
});
