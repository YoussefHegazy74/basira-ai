const CITATION_RE = /\(?\s*مصدر\s*(\d+)\s*\)?/g;

const SOURCE_TYPE_AR = {
  quran: "القرآن",
  hadith: "حديث",
  tafsir: "تفسير",
  fiqh: "فقه",
  sira: "سيرة",
};

function sourceReference(source) {
  if (typeof source === "string") return source.trim();
  if (!source || typeof source !== "object") return "";
  const reference = String(
    source.reference || source.title || source.name || source.url || "",
  ).trim();
  const type = String(source.type || source.source || "").trim();
  if (type && reference && !reference.toLowerCase().startsWith(type.toLowerCase())) {
    return `${type} — ${reference}`;
  }
  return reference;
}

export function formatSourceLabel(source) {
  const reference = sourceReference(source);
  if (!reference) return "";
  const parts = reference.split(/\s+[—–-]\s+/);
  if (parts.length < 2) return reference;
  const type = parts.shift().trim().toLowerCase();
  const label = SOURCE_TYPE_AR[type] || type;
  return `${label} — ${parts.join(" — ").trim()}`;
}

export function citationSegments(text, sources) {
  const value = String(text || "");
  const list = Array.isArray(sources) ? sources : [];
  const segments = [];
  let cursor = 0;
  let match;
  CITATION_RE.lastIndex = 0;

  while ((match = CITATION_RE.exec(value)) !== null) {
    if (match.index > cursor) {
      segments.push({ type: "text", text: value.slice(cursor, match.index) });
    }
    const index = Number(match[1]);
    const label = index > 0 && index <= list.length
      ? formatSourceLabel(list[index - 1])
      : "";
    if (label) {
      segments.push({
        type: "citation",
        text: `[${label}]`,
        index,
      });
    }
    cursor = match.index + match[0].length;
  }

  if (cursor < value.length) {
    segments.push({ type: "text", text: value.slice(cursor) });
  }
  return segments;
}

export function citedSourceLabels(text, sources) {
  const list = Array.isArray(sources) ? sources : [];
  const seen = new Set();
  const labels = [];
  for (const segment of citationSegments(text, list)) {
    if (segment.type !== "citation" || seen.has(segment.index)) continue;
    seen.add(segment.index);
    labels.push(segment.text.slice(1, -1));
  }
  return labels;
}
