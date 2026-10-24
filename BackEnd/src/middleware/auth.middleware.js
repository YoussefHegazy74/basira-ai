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
        new AppError( 401, "أنت غير مسجل الدخول", HttpStatusText.FAIL),
      );
    }

    const decoded = await promisify(jwt.verify)(token, process.env.JWT_SECRET);
    const currentUser = await User.findById(decoded.id);

    if (!currentUser) {
      return next(
        new AppError( 401, "المستخدم لم يعد موجودًا", HttpStatusText.FAIL),
      );
    }

    if (currentUser.changedPasswordAfter(decoded.iat)) {
      return next(
        new AppError(
          401,
          "قام المستخدم بتغيير كلمة المرور مؤخرًا! يرجى تسجيل الدخول مرة أخرى.",
          HttpStatusText.FAIL,
        ),
      );
    }

    req.user = currentUser;
    next();
  } catch (err) {
    next(new AppError( 401, "رمز غير صالح", HttpStatusText.FAIL));
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
        new AppError( 401, "المستخدم لم يعد موجودًا", HttpStatusText.FAIL),
      );
    }

    if (currentUser.changedPasswordAfter(decoded.iat)) {
      return next(
        new AppError(
          401,
          "قام المستخدم بتغيير كلمة المرور مؤخرًا! يرجى تسجيل الدخول مرة أخرى.",
          HttpStatusText.FAIL,
        ),
      );
    }

    req.user = currentUser;
    next();
  } catch (err) {
    next(new AppError( 401, "رمز غير صالح", HttpStatusText.FAIL));
  }
};

module.exports = { protect, optionalAuth };
