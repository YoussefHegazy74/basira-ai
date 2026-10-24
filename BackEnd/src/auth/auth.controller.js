const bcrypt = require("bcrypt");
const jwt = require("jsonwebtoken");
const crypto = require("crypto");
const AppError = require("../utils/AppError.js");
const HttpStatus = require("../utils/HttpStatusText.js");
const AuthService = require("./auth.service.js");
const asyncWrapper = require("../middleware/Async_wrapper.js");
const register = asyncWrapper(async (req, res, next) => {
  await AuthService.register(req.body);
  res.status(201).json({
    status: HttpStatus.SUCCESS,
    Message: "تحقق من بريدك الإلكتروني للحصول على رمز التحقق.",
  });
});
const login = asyncWrapper(async (req, res, next) => {
  const { token, user } = await AuthService.login(req.body);

  res.cookie("token", token, {
    httpOnly: true,
    sameSite: "lax",
    maxAge: 7 * 24 * 60 * 60 * 1000,
    secure: true,
  });

  res.status(200).json({
    status: HttpStatus.SUCCESS,
    user,
  });
});

const forgotPassword = asyncWrapper(async (req, res, next) => {
  await AuthService.forgotPassword(req, res);
  res.status(200).json({
    status: HttpStatus.SUCCESS,
    message: "إذا كان البريد الإلكتروني موجودًا، فقد تم إرسال رابط إعادة تعيين كلمة المرور.",
  });
});
const resetPassword = asyncWrapper(async (req, res, next) => {
  await AuthService.resetPassword(
    req.params.token,
    req.body.newPassword,
    req.body.newPasswordConfirm,
  );
  res.status(200).json({
    status: HttpStatus.SUCCESS,
    message: "تمت إعادة تعيين كلمة المرور بنجاح.",
  });
});
const changePassword = asyncWrapper(async (req, res, next) => {
  const { token } = await AuthService.changePassword(
    req.user._id,
    req.body.currentPassword,
    req.body.newPassword,
  );
  res.cookie("token", token, {
    httpOnly: true,
    sameSite: "lax",
    maxAge: 7 * 24 * 60 * 60 * 1000,
    secure: true,
  });
  res.status(200).json({
    status: HttpStatus.SUCCESS,
    message: "تم تغيير كلمة المرور بنجاح.",
  });
});
const verifyEmail = asyncWrapper(async (req, res, next) => {
  await AuthService.verifyEmail(
    String(req.body.verificationToken),
    req.body.email,
  );
  res.status(200).json({
    status: HttpStatus.SUCCESS,
    message: "تم التحقق من البريد الإلكتروني بنجاح.",
  });
});
const logout = asyncWrapper(async (req, res, next) => {
  await AuthService.logout(req, res);
  res.status(200).json({ status: HttpStatus.SUCCESS, message: "تم تسجيل الخروج" });
});
/// deactivate account
const deactivateAccount = asyncWrapper(async (req, res, next) => {
  await AuthService.deactivateAccount(req.user._id);
  res.status(200).json({
    status: HttpStatus.SUCCESS,
    message: "تم تعطيل الحساب بنجاح.",
  });
});
module.exports = {
  register,
  login,
  forgotPassword,
  resetPassword,
  changePassword,
  verifyEmail,
  logout,
  deactivateAccount,
};
