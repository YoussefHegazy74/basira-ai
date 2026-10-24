import { test } from "node:test";
import assert from "node:assert/strict";

import {
  activeConversationKey,
  buildChatHistory,
  makeClientMessageId,
  mapStoredMessages,
  mergeServerMessages,
  readActiveConversation,
  readFailedSubmission,
  writeActiveConversation,
  writeFailedSubmission,
} from "../src/utils/chatHistory.js";

function memoryStorage() {
  const data = new Map();
  return {
    getItem: (key) => (data.has(key) ? data.get(key) : null),
    setItem: (key, value) => data.set(key, String(value)),
    removeItem: (key) => data.delete(key),
  };
}

test("active conversation is stored per user and survives a reload", () => {
  const storage = memoryStorage();
  const userOne = { id: "user-one" };
  const userTwo = { id: "user-two" };

  writeActiveConversation(userOne, "conversation-a", storage);
  assert.equal(readActiveConversation(userOne, storage), "conversation-a");
  assert.equal(readActiveConversation(userTwo, storage), null);
  assert.equal(
    activeConversationKey(userOne),
    "basira.activeConversation.user-one",
  );

  writeActiveConversation(userOne, null, storage);
  assert.equal(readActiveConversation(userOne, storage), null);
});

test("retryable failed submission survives a reload per user", () => {
  const storage = memoryStorage();
  const user = { id: "user-one" };
  const failed = {
    text: "question",
    clientMessageId: "client-1",
    conversationId: "conversation-1",
  };

  writeFailedSubmission(user, failed, storage);
  assert.deepEqual(readFailedSubmission(user, storage), failed);
  writeFailedSubmission(user, null, storage);
  assert.equal(readFailedSubmission(user, storage), null);
});

test("history excludes the current optimistic message and keeps four turns", () => {
  const messages = [
    { sender: "system", text: "ignored" },
    { sender: "user", text: "u1" },
    { sender: "assistant", text: "a1" },
    { sender: "user", text: "u2" },
    { sender: "assistant", text: "a2" },
    { sender: "user", text: "u3" },
    { sender: "assistant", text: "a3" },
    { sender: "user", text: "current", clientMessageId: "current-id" },
  ];

  assert.deepEqual(buildChatHistory(messages, "current-id"), [
    { role: "user", content: "u2" },
    { role: "assistant", content: "a2" },
    { role: "user", content: "u3" },
    { role: "assistant", content: "a3" },
  ]);
});

test("stored messages map to renderer state without losing sources", () => {
  const mapped = mapStoredMessages([
    {
      _id: "m1",
      role: "assistant",
      content: "answer",
      sources: ["fiqh — source"],
    },
  ]);
  assert.deepEqual(mapped, [
    {
      id: "m1",
      text: "answer",
      sender: "assistant",
      sources: ["fiqh — source"],
    },
  ]);
});

test("server response reconciles optimistic user and assistant exactly once", () => {
  const current = [
    {
      id: "local-user-id",
      clientMessageId: "client-1",
      text: "question",
      sender: "user",
    },
  ];
  const data = {
    conversation: { _id: "conversation-1" },
    userMessage: { _id: "server-user-id", content: "question" },
    assistantMessage: {
      _id: "server-assistant-id",
      content: "answer",
      sources: ["source"],
    },
  };

  const first = mergeServerMessages(current, data, "client-1");
  assert.deepEqual(first, [
    {
      id: "server-user-id",
      clientMessageId: "client-1",
      text: "question",
      sender: "user",
    },
    {
      id: "server-assistant-id",
      text: "answer",
      sender: "assistant",
      sources: ["source"],
    },
  ]);

  const replayed = mergeServerMessages(first, data, "client-1");
  assert.equal(replayed.length, 2);
});

test("retry can restore a user message when the optimistic bubble is gone", () => {
  const data = {
    userMessage: { _id: "server-user-id", content: "question" },
    assistantMessage: { _id: "server-assistant-id", content: "answer" },
  };
  const merged = mergeServerMessages([], data, "client-1", {
    appendUser: true,
  });
  assert.deepEqual(
    merged.map((item) => [item.sender, item.text]),
    [
      ["user", "question"],
      ["assistant", "answer"],
    ],
  );
});

test("client message IDs are non-empty and unique", () => {
  const first = makeClientMessageId();
  const second = makeClientMessageId();
  assert.equal(typeof first, "string");
  assert.ok(first.length > 8);
  assert.notEqual(first, second);
});
