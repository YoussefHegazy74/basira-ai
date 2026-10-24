const express = require("express");
const mongoose = require("mongoose");
const app = express();

// Authenticated API responses must never be served from a browser cache.
// Conditional GETs were returning 304 with no JSON body when a conversation
// was reopened, which made the Chat History view fail to hydrate.
app.set("etag", false);
app.use((req, res, next) => {
  res.set("Cache-Control", "no-store");
  next();
});

// Liveness/readiness are intentionally mounted before the API rate limiter so
// container and reverse-proxy probes do not consume user request quota.
app.get("/health", (req, res) => {
  res.status(200).json({ status: "ok" });
});
app.get("/ready", (req, res) => {
  const ready = mongoose.connection.readyState === 1;
  res.status(ready ? 200 : 503).json({
    status: ready ? "ready" : "not-ready",
  });
});

// ===== Third-Party Middleware =====
const cors = require("cors");
const { rateLimit } = require("express-rate-limit");
const helmet = require("helmet");
const mongosanitize = require("express-mongo-sanitize");
const xss = require("xss-clean");
const cookieParser = require("cookie-parser");
const morgan = require("morgan");

// ===== Routes =====
const authRoutes = require("./auth/auth.routes.js");
const conversationRoutes = require("./conversation/conversation.routes.js");
const messageRoutes = require("./message/message.routes.js");
const usageRoutes = require("./usage/usage.routes.js");
const tafsirRoutes = require("./tafsir/tafsir.routes.js");
const verifyRoutes = require("./verify/verify.routes.js");
const userRoutes = require("./user/user.routes.js");
// ===== Utils =====
const HttpStatusText = require("./utils/HttpStatusText.js");

// ===== Rate Limiter Config =====
const limiter = rateLimit({
  windowMs: 60 * 60 * 1000, // 1 hour
  max: 100,
  message: "Too many requests from this IP, please try again after 1 hour",
});

// ===== Global Middleware =====
const allowedOrigins = new Set(
  [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    process.env.FRONTEND_URL,
    ...(process.env.CORS_ORIGINS || "").split(","),
  ]
    .map((origin) => (origin || "").trim())
    .filter(Boolean),
);

app.use(cookieParser());
app.use(helmet());
app.use(limiter);
app.use(
  cors({
    // Reflecting an arbitrary Origin with credentials enabled would let any
    // site call authenticated endpoints.  Use an explicit allowlist; deploy
    // targets add their origin through FRONTEND_URL or CORS_ORIGINS.
    origin(origin, callback) {
      if (!origin || allowedOrigins.has(origin)) {
        return callback(null, true);
      }
      return callback(null, false);
    },
    credentials: true,
  }),
);
app.use(morgan("dev"));
app.use(express.json());

// Make req.query writable
app.use((req, res, next) => {
  Object.defineProperty(req, "query", {
    value: { ...req.query },
    writable: true,
    configurable: true,
    enumerable: true,
  });

  next();
});

// ===== Input Sanitization =====
app.use(mongosanitize());
app.use(xss());

// ===== Routes =====
app.use("/auth", authRoutes);
app.use("/conversations", conversationRoutes);
app.use("/messages", messageRoutes);
app.use("/usage", usageRoutes);
app.use("/tafsir", tafsirRoutes);
app.use("/verify", verifyRoutes);
app.use("/users", userRoutes);

// ===== 404 Handler =====
app.use((req, res, next) => {
  return res.status(404).json({
    status: HttpStatusText.FAIL,
    message: "Page Not Found",
  });
});

// ===== Global Error Handler =====
app.use((err, req, res, next) => {
  // Mongoose validation error
  if (err.name === "ValidationError") {
    return res.status(400).json({
      status: HttpStatusText.FAIL,
      errName: err.name,
      message: Object.values(err.errors)
        .map((el) => el.message)
        .join(", "),
    });
  }

  // JWT errors
  if (err.name === "JsonWebTokenError" || err.name === "TokenExpiredError") {
    return res.status(401).json({
      status: HttpStatusText.FAIL,
      message: "Invalid or expired token, please log in again",
    });
  }

  // Duplicate key error (MongoDB)
  if (err.code === 11000) {
    return res.status(400).json({
      status: HttpStatusText.FAIL,
      message: "Email already exists",
    });
  }

  // Default error
  const payload = {
    status: err.statustext || HttpStatusText.ERROR,
    message: err.message || "Something Went Wrong",
    errors: err.errors || [],
  };
  // Structured recovery metadata (conversation/message IDs, retryable flag)
  // must reach the browser so a failed generation can be retried idempotently.
  if (err.data && typeof err.data === "object") payload.data = err.data;
  res.status(err.statusCode || 500).json(payload);
});

module.exports = app;
