const { test } = require("node:test");
const assert = require("node:assert/strict");
const http = require("node:http");
const {
  askModel,
  verify,
} = require("../src/verify/verify.service.js");

const TEXT = "ولا تقربوا الصلاة";
const EXPLANATION = "الآية تنهى عن الصلاة في حال السكر.";

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

test("askModel sends the verification contract and returns structured data", async (t) => {
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
      res.end(JSON.stringify({
        verification_status: "misleading",
        overall_status: "misleading",
        analysis_mode: "post",
        component_count: 1,
        status_label: "تحذير — اقتباس أو استدلال مضلّل",
        text_status: "صحيح",
        quotation: {
          found: true,
          reference: "النساء:43",
          is_partial: true,
          canonical_text: "يَا أَيُّهَا الَّذِينَ آمَنُوا...",
        },
        interpretation: {
          provided: true,
          status: "contradicted",
          label: "الشرح غير مدعوم بالمصدر",
        },
        components: [{
          id: "c1",
          type: "quran_quote",
          source_identity: { status: "partial" },
          completeness: { status: "partial_material", is_material: true },
          authenticity: { status: "not_applicable" },
          interpretation: { status: "contradicted" },
          argument: { status: "misleading" },
        }],
      }));
    });
  });

  const previousUrl = process.env.AI_URL;
  process.env.AI_URL = url;
  t.after(async () => {
    process.env.AI_URL = previousUrl;
    await closeServer(server);
  });

  const result = await askModel(TEXT, EXPLANATION);

  assert.equal(received.method, "POST");
  assert.equal(received.url, "/api/verify");
  assert.equal(received.contentType, "application/json");
  assert.deepEqual(received.body, {
    text: TEXT,
    explanation: EXPLANATION,
  });
  assert.equal(result.verification_status, "misleading");
  assert.equal(result.analysis_mode, "post");
  assert.equal(result.component_count, 1);
  assert.equal(result.quotation.reference, "النساء:43");
  assert.equal(result.interpretation.status, "contradicted");
  assert.equal(result.components[0].argument.status, "misleading");
});

test("askModel maps an unreachable verification service to a safe 502", async (t) => {
  const previousUrl = process.env.AI_URL;
  process.env.AI_URL = "http://127.0.0.1:1";

  try {
    await assert.rejects(
      () => askModel(TEXT),
      (error) => {
        assert.equal(error.statusCode, 502);
        assert.match(error.message, /تعذّر إكمال التحقق/);
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
    res.end(JSON.stringify({
      detail: "OPENAI_API_KEY=secret-value internal stack trace",
    }));
  });

  const previousUrl = process.env.AI_URL;
  process.env.AI_URL = url;
  t.after(async () => {
    process.env.AI_URL = previousUrl;
    await closeServer(server);
  });

  await assert.rejects(
    () => askModel(TEXT),
    (error) => {
      assert.equal(error.statusCode, 502);
      assert.match(error.message, /تعذّر إكمال التحقق/);
      assert.doesNotMatch(error.message, /OPENAI|secret|stack trace/i);
      return true;
    },
  );
});

test("empty verification text is rejected before any service call", async () => {
  await assert.rejects(
    () => verify("", "", "user-id", "guest-key"),
    (error) => {
      assert.equal(error.statusCode, 400);
      assert.match(error.message, /النص مطلوب/);
      return true;
    },
  );
});
