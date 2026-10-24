const restrictTo = (...roles) => {
  return (req, res, next) => {
    if (!roles.includes(req.user.role)) {
      return res.status(403).json({
        status: "fail",
        message: "ليس لديك صلاحية لتنفيذ هذا الإجراء.",
      });
    }

    next();
  };
};

module.exports = restrictTo;
