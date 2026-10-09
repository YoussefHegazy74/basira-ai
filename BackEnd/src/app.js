const express = require("express");
const app = express();
const authRoutes = require("./auth/auth.routes.js");
const HttpStatusText = require("./utils/HttpStatusText.js");
const cors = require("cors");
const { rateLimit } = require("express-rate-limit");
const morgan = require("morgan");
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100,
  message: "Too many requests from this IP, please try again after 15 minutes",
});
app.use(limiter);
app.use(cors());
app.use(morgan("dev"));
app.use(express.json());
app.use("/auth", authRoutes);

app.use((req, res, next) => {
  return res.status(404).json({
    status: HttpStatusText.FAIL,
    message: "Page Not Found",
  });
});
app.use((err, req, res, next) => {
  let errors = [];

  if (err.name === "ValidationError") {
    return res.status(400).json({
      status: "FAIL",
      errName: err.name,
      message: Object.values(err.errors)
        .map((el) => el.message)
        .join(", "),
    });
  }
  if (err.name === "JsonWebTokenError" || err.name === "TokenExpiredError") {
    return res.status(401).json({
      status: HttpStatusText.FAIL,
      message: "Invalid or expired token, please log in again",
    });
  }
  if (err.code === 11000) {
    return res.status(400).json({
      status: HttpStatusText.FAIL,
      message: "Email already exists",
    });
  }
  res.status(err.statusCode || 500).json({
    status: err.statustext || HttpStatusText.ERROR,
    message: err.message || "Something Went Wrong",
    errors: err.errors || [],
  });
});
module.exports = app;
