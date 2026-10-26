const LOOPBACK_HOSTS = new Set(["localhost", "127.0.0.1", "[::1]", "::1"]);

export function alignLoopbackApiUrl(value, pageHostname) {
  if (typeof value !== "string" || !/^https?:\/\//i.test(value)) {
    return value;
  }

  try {
    const url = new URL(value);
    if (
      LOOPBACK_HOSTS.has(url.hostname) &&
      LOOPBACK_HOSTS.has(pageHostname)
    ) {
      // A cookie set for localhost is not sent on cross-site requests from
      // 127.0.0.1 (and vice versa). Keep local development on the same host
      // as the page so the SameSite=Lax session cookie is actually sent.
      url.hostname = pageHostname;
    }
    return url.toString().replace(/\/$/, "");
  } catch {
    return value;
  }
}
