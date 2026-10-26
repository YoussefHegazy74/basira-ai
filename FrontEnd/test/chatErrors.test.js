import { test } from "node:test";
import assert from "node:assert/strict";

import {
  CHAT_CONNECTION_ERROR,
  CHAT_SERVER_ERROR,
  chatRequestErrorMessage,
} from "../src/utils/chatErrors.js";

test("a network failure keeps the connection message", () => {
  assert.equal(
    chatRequestErrorMessage({ message: "Network Error" }),
    CHAT_CONNECTION_ERROR,
  );
});

test("the Arabic conversation-limit message is shown verbatim", () => {
  const message =
    "تم الوصول إلى الحد الأقصى لعدد الرسائل في هذه المحادثة. " +
    "ابدأ محادثة جديدة للمتابعة.";
  assert.equal(
    chatRequestErrorMessage({ response: { status: 403, data: { message } } }),
    message,
  );
});

test("a server response with a non-Arabic message is not called a network error", () => {
  assert.equal(
    chatRequestErrorMessage({
      response: { status: 500, data: { message: "Internal server error" } },
    }),
    CHAT_SERVER_ERROR,
  );
});

test("a server response without a message uses the server fallback", () => {
  assert.equal(
    chatRequestErrorMessage({ response: { status: 502, data: {} } }),
    CHAT_SERVER_ERROR,
  );
});
