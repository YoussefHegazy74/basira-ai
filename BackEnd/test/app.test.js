const { test, before, after } = require("node:test");
const assert = require("node:assert/strict");
const app = require("../src/app");

let server;
let baseUrl;

before(async () => {
  server = app.listen(0, "127.0.0.1");
  await new Promise((resolve, reject) => {
    server.once("listening", resolve);
    server.once("error", reject);
  });
  baseUrl = `http://127.0.0.1:${server.address().port}`;
});

after(async () => {
  await new Promise((resolve) => server.close(resolve));
});

test("unknown route returns the JSON 404 contract", async () => {
  const response = await fetch(`${baseUrl}/__missing__`);

  assert.equal(response.status, 404);
  assert.deepEqual(await response.json(), {
    status: "Fail",
    message: "Page Not Found",
  });
});

test("CORS preflight reflects the frontend origin with credentials", async () => {
  const origin = "http://localhost:5173";
  const response = await fetch(`${baseUrl}/messages`, {
    method: "OPTIONS",
    headers: {
      Origin: origin,
      "Access-Control-Request-Method": "POST",
      "Access-Control-Request-Headers": "content-type",
    },
  });

  assert.equal(response.status, 204);
  assert.equal(response.headers.get("access-control-allow-origin"), origin);
  assert.equal(response.headers.get("access-control-allow-credentials"), "true");
});

test("guest chat rejects an empty message before touching MongoDB", async () => {
  const response = await fetch(`${baseUrl}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });

  assert.equal(response.status, 400);
  assert.deepEqual(await response.json(), {
    status: "Bad Request",
    message: "Message is required",
    errors: [],
  });
});

test("tafsir rejects a missing ayah before touching MongoDB", async () => {
  const response = await fetch(`${baseUrl}/tafsir`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });

  assert.equal(response.status, 400);
  assert.deepEqual(await response.json(), {
    status: "Bad Request",
    message: "Ayah is required",
    errors: [],
  });
});
