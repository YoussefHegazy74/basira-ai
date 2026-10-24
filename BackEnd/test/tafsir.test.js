const { test } = require("node:test");
const assert = require("node:assert/strict");
const http = require("node:http");
const { askModel } = require("../src/tafsir/tafsir.service.js");

const AYAH = "قل هو الله أحد";

const startServer = async (handler) => {
  const server = http.createServer(handler);
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  return {
    server,
    url: `http://127.0.0.1:${server.address().port}`,
  };
};

const closeServer = (server) =>
  new Promise((resolve) => server.close(resolve));

test("askModel sends the RAG tafsir contract and returns its response", async (t) => {
  let received;
  const { server, url } = await startServer((req, res) => {
    let raw = "";
    req.on("data", (chunk) => {
      raw += chunk;
    });
    req.on("end", () => {
      received = {
        method: req.method,
        url: req.url,
        contentType: req.headers["content-type"],
        body: JSON.parse(raw),
      };
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(
        JSON.stringify({
          tafsir: "تفسير موثق",
          context: "",
          source: "112:1",
          input_ayah: AYAH,
        }),
      );
    });
  });

  const previousUrl = process.env.AI_URL;
  process.env.AI_URL = url;
  t.after(async () => {
    process.env.AI_URL = previousUrl;
    await closeServer(server);
  });

  const result = await askModel(AYAH);

  assert.equal(received.method, "POST");
  assert.equal(received.url, "/api/tafsir");
  assert.equal(received.contentType, "application/json");
  assert.deepEqual(received.body, { ayah: AYAH });
  assert.equal(result.source, "112:1");
  assert.equal(result.input_ayah, AYAH);
});

test("askModel maps an unreachable RAG service to a safe 502", async () => {
  const previousUrl = process.env.AI_URL;
  process.env.AI_URL = "http://127.0.0.1:1";

  try {
    await assert.rejects(
      () => askModel(AYAH),
      (error) => {
        assert.equal(error.statusCode, 502);
        assert.match(error.message, /Tafsir service is unavailable/i);
        return true;
      },
    );
  } finally {
    process.env.AI_URL = previousUrl;
  }
});

test("askModel does not expose upstream failure details", async (t) => {
  const { server, url } = await startServer((req, res) => {
    res.writeHead(503, { "Content-Type": "application/json" });
    res.end(
      JSON.stringify({
        detail: "OPENAI_API_KEY=secret-value internal stack trace",
      }),
    );
  });

  const previousUrl = process.env.AI_URL;
  process.env.AI_URL = url;
  t.after(async () => {
    process.env.AI_URL = previousUrl;
    await closeServer(server);
  });

  await assert.rejects(
    () => askModel(AYAH),
    (error) => {
      assert.equal(error.statusCode, 502);
      assert.match(error.message, /Tafsir service is unavailable/i);
      assert.doesNotMatch(error.message, /OPENAI|secret|stack trace/i);
      return true;
    },
  );
});

test("askModel maps a RAG timeout to a safe 504", async () => {
  const originalFetch = global.fetch;
  global.fetch = async () => {
    const error = new Error("upstream timed out");
    error.name = "TimeoutError";
    throw error;
  };

  try {
    await assert.rejects(
      () => askModel(AYAH),
      (error) => {
        assert.equal(error.statusCode, 504);
        assert.match(error.message, /timed out/i);
        return true;
      },
    );
  } finally {
    global.fetch = originalFetch;
  }
});
