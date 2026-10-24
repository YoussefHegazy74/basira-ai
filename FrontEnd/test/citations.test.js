import { test } from "node:test";
import assert from "node:assert/strict";

import {
  citationSegments,
  citedSourceLabels,
  formatSourceLabel,
} from "../src/utils/citations.js";

const SOURCES = [
  "hadith — سنن أبي داود:54",
  "quran — المائدة:50",
  "fiqh — فقه السنة، سيد سابق، الطهارة - سنن الفطرة (2/3)",
];

test("internal source marker becomes a readable inline reference", () => {
  const segments = citationSegments(
    "حكم اللحية هو الإعفاء (مصدر 1).",
    SOURCES,
  );
  const citations = segments.filter((segment) => segment.type === "citation");
  assert.equal(citations.length, 1);
  assert.equal(citations[0].text, "[حديث — سنن أبي داود:54]");
  assert.ok(!segments.some((segment) => segment.text.includes("مصدر")));
});

test("only cited sources are returned, in citation order", () => {
  const text = "الدليل في الحديث (مصدر 1)، ويؤيده القرآن (مصدر 2).";
  assert.deepEqual(citedSourceLabels(text, SOURCES), [
    "حديث — سنن أبي داود:54",
    "القرآن — المائدة:50",
  ]);
  assert.ok(!citedSourceLabels(text, SOURCES).includes("فقه — فقه السنة، سيد سابق، الطهارة - سنن الفطرة (2/3)"));
});

test("invalid source index is not exposed to the user", () => {
  const segments = citationSegments("إجابة بلا مرجع صالح (مصدر 9).", SOURCES);
  assert.ok(!segments.some((segment) => segment.type === "citation"));
  assert.ok(!segments.some((segment) => segment.text.includes("مصدر")));
  assert.deepEqual(citedSourceLabels("إجابة (مصدر 9).", SOURCES), []);
});

test("duplicate markers are deduplicated in the source summary", () => {
  const text = "الأول (مصدر 1) والثاني (مصدر 1) والثالث (مصدر 2).";
  const segments = citationSegments(text, SOURCES);
  assert.equal(
    segments.filter((segment) => segment.type === "citation").length,
    3,
  );
  assert.deepEqual(citedSourceLabels(text, SOURCES), [
    "حديث — سنن أبي داود:54",
    "القرآن — المائدة:50",
  ]);
});

test("plain conversational answer passes through unchanged", () => {
  const text = "نعم، الإعفاء هو الحكم الذي تدعمه المصادر.";
  const segments = citationSegments(text, SOURCES);
  assert.deepEqual(segments, [{ type: "text", text }]);
  assert.deepEqual(citedSourceLabels(text, SOURCES), []);
});

test("empty or malformed input is safe", () => {
  assert.deepEqual(citationSegments("", SOURCES), []);
  assert.deepEqual(citationSegments(null, null), []);
  assert.deepEqual(citedSourceLabels(undefined, SOURCES), []);
});

test("source labels support object metadata and unknown types", () => {
  assert.equal(
    formatSourceLabel({ source: "hadith", reference: "صحيح مسلم:507" }),
    "حديث — صحيح مسلم:507",
  );
  assert.equal(
    formatSourceLabel({ reference: "كتاب مبهم، ص 4" }),
    "كتاب مبهم، ص 4",
  );
  assert.equal(formatSourceLabel({}), "");
});
