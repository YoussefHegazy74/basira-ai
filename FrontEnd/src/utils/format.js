const numberFormatter = new Intl.NumberFormat("ar-EG", {
  useGrouping: false,
});

const groupedNumberFormatter = new Intl.NumberFormat("ar-EG");

export function formatNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? numberFormatter.format(number) : "—";
}

export function formatCount(value) {
  const number = Number(value);
  return Number.isFinite(number) ? groupedNumberFormatter.format(number) : "—";
}

export function formatDate(value, options) {
  if (!value) return "—";
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return new Intl.DateTimeFormat("ar-EG", {
    day: "numeric",
    month: "long",
    year: "numeric",
    ...options,
  }).format(date);
}

export function formatYear(value = new Date()) {
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return new Intl.DateTimeFormat("ar-EG", { year: "numeric" }).format(date);
}

export function arabicMessage(value, fallback) {
  const text = typeof value === "string" ? value.trim() : "";
  return /[\u0600-\u06FF]/.test(text) ? text : fallback;
}

export function initials(name, fallback = "ب") {
  const parts = String(name || "")
    .trim()
    .split(/\s+/)
    .filter(Boolean);
  if (!parts.length) return fallback;
  return parts
    .slice(0, 2)
    .map((part) => part.charAt(0))
    .join("");
}
