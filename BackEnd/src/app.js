const express = require("express");
const app = express();

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
app.use(cookieParser());
app.use(helmet());
app.use(limiter);
app.use(cors());
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
  res.status(err.statusCode || 500).json({
    status: err.statustext || HttpStatusText.ERROR,
    message: err.message || "Something Went Wrong",
    errors: err.errors || [],
  });
});

module.exports = app;
