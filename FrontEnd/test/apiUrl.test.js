import { test } from "node:test";
import assert from "node:assert/strict";

import { alignLoopbackApiUrl } from "../src/utils/apiUrl.js";

test("localhost API is aligned to a 127.0.0.1 frontend", () => {
  assert.equal(
    alignLoopbackApiUrl("http://localhost:3000", "127.0.0.1"),
    "http://127.0.0.1:3000",
  );
});

test("127.0.0.1 API is aligned to a localhost frontend", () => {
  assert.equal(
    alignLoopbackApiUrl("http://127.0.0.1:3000", "localhost"),
    "http://localhost:3000",
  );
});

test("non-loopback production API URLs are unchanged", () => {
  assert.equal(
    alignLoopbackApiUrl("https://api.basira.example", "localhost"),
    "https://api.basira.example",
  );
});

test("relative or malformed API values are unchanged", () => {
  assert.equal(alignLoopbackApiUrl("/api", "127.0.0.1"), "/api");
  assert.equal(alignLoopbackApiUrl("not a url", "127.0.0.1"), "not a url");
});

test("matching loopback hosts are left unchanged", () => {
  assert.equal(
    alignLoopbackApiUrl("http://localhost:3000", "localhost"),
    "http://localhost:3000",
  );
  assert.equal(
    alignLoopbackApiUrl("http://127.0.0.1:3000", "127.0.0.1"),
    "http://127.0.0.1:3000",
  );
});
