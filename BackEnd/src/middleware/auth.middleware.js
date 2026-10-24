const jwt = require("jsonwebtoken");
const { promisify } = require("util");
const User = require("../user/user.model.js");
const AppError = require("../utils/AppError.js");
const HttpStatusText = require("../utils/HttpStatusText.js");

// ===== Get token from cookie OR Authorization header =====
const getTokenFromRequest = (req) => {
  // 1) From cookie
  if (req.cookies?.token) return req.cookies.token;

  // 2) From Bearer header
  if (
    req.headers.authorization &&
    req.headers.authorization.startsWith("Bearer")
  ) {
    return req.headers.authorization.split(" ")[1];
  }

  return null;
};

// ===== Required auth =====
const protect = async (req, res, next) => {
  try {
    const token = getTokenFromRequest(req);

    if (!token) {
      return next(
        new AppError( 401, "You are not logged in", HttpStatusText.FAIL),
      );
    }

    const decoded = await promisify(jwt.verify)(token, process.env.JWT_SECRET);
    const currentUser = await User.findById(decoded.id);

    if (!currentUser) {
      return next(
        new AppError( 401, "User no longer exists", HttpStatusText.FAIL),
      );
    }

    if (!currentUser.isActive) {
      return next(
        new AppError(401, "Account is deactivated", HttpStatusText.FAIL),
      );
    }

    if (currentUser.changedPasswordAfter(decoded.iat)) {
      return next(
        new AppError(
          401,
          "User recently changed password! Please log in again.",
          HttpStatusText.FAIL,
        ),
      );
    }

    req.user = currentUser;
    next();
  } catch (err) {
    next(new AppError( 401, "Invalid token", HttpStatusText.FAIL));
  }
};

// ===== Optional auth: no token -> guest, bad token -> 401 =====
const optionalAuth = async (req, res, next) => {
  try {
    const token = getTokenFromRequest(req);

    if (!token) return next();

    const decoded = await promisify(jwt.verify)(token, process.env.JWT_SECRET);
    const currentUser = await User.findById(decoded.id);

    if (!currentUser) {
      return next(
        new AppError( 401, "User no longer exists", HttpStatusText.FAIL),
      );
    }

    if (!currentUser.isActive) {
      return next(
        new AppError(401, "Account is deactivated", HttpStatusText.FAIL),
      );
    }

    if (currentUser.changedPasswordAfter(decoded.iat)) {
      return next(
        new AppError(
          401,
          "User recently changed password! Please log in again.",
          HttpStatusText.FAIL,
        ),
      );
    }

    req.user = currentUser;
    next();
  } catch (err) {
    next(new AppError( 401, "Invalid token", HttpStatusText.FAIL));
  }
};

module.exports = { protect, optionalAuth };
