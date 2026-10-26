import { test } from "node:test";
import assert from "node:assert/strict";

import {
  LOGIN_CONNECTION_ERROR,
  LOGIN_CREDENTIALS_ERROR,
  LOGIN_RATE_LIMIT_FALLBACK,
  LOGIN_SERVER_ERROR,
  loginErrorMessage,
} from "../src/utils/authErrors.js";

test("invalid credentials show the backend Arabic message", () => {
  assert.equal(
    loginErrorMessage({
      response: {
        status: 401,
        data: { status: "Fail", message: "البريد الإلكتروني أو كلمة المرور غير صحيحة" },
      },
    }),
    "البريد الإلكتروني أو كلمة المرور غير صحيحة",
  );
});

test("the plain-text rate-limit response is shown instead of a credentials error", () => {
  assert.equal(
    loginErrorMessage({
      response: {
        status: 429,
        data: "طلبات كثيرة جدًا من هذا العنوان، يرجى المحاولة بعد ساعة",
      },
    }),
    "طلبات كثيرة جدًا من هذا العنوان، يرجى المحاولة بعد ساعة",
  );
});

test("a rate-limit response without a usable message is not called invalid credentials", () => {
  assert.equal(
    loginErrorMessage({ response: { status: 429, data: "" } }),
    LOGIN_RATE_LIMIT_FALLBACK,
  );
});

test("missing response is a connection error", () => {
  assert.equal(
    loginErrorMessage({ message: "Network Error" }),
    LOGIN_CONNECTION_ERROR,
  );
});

test("server-side errors are not presented as credentials errors", () => {
  assert.equal(
    loginErrorMessage({
      response: { status: 500, data: { message: "Internal server error" } },
    }),
    LOGIN_SERVER_ERROR,
  );
  assert.notEqual(LOGIN_SERVER_ERROR, LOGIN_CREDENTIALS_ERROR);
});
