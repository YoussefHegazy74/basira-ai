const bcrypt = require("bcrypt");
const jwt = require("jsonwebtoken");
const crypto = require("crypto");
const AppError = require("../utils/appError.js");
const HttpStatus = require("../utils/HttpStatusText.js");
const AuthService = require("./auth.service.js");
const asyncWrapper = require("../middleware/Async_wrapper.js");
const register = asyncWrapper(async (req, res, next) => {
  await AuthService.register(req.body);
  res.status(201).json({
    status: HttpStatus.SUCCESS,
    Message: "Check your email for verification code.",
  });
});
const login = asyncWrapper(async (req, res, next) => {
  const { token } = await AuthService.login(req.body);
  res.status(200).json({ status: HttpStatus.SUCCESS, token });
});
const forgotPassword = asyncWrapper(async (req, res, next) => {
  await AuthService.forgotPassword(req, res);
  res.status(200).json({
    status: HttpStatus.SUCCESS,
    message: "If the email exists, a password reset link has been sent.",
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
    message: "Password has been reset successfully.",
  });
});
const changePassword = asyncWrapper(async (req, res, next) => {
  const { token } = await AuthService.changePassword(
    req.user._id,
    req.body.currentPassword,
    req.body.newPassword,
  );
  res.status(200).json({
    status: HttpStatus.SUCCESS,
    message: "Password has been changed successfully.",
    token,
  });
});
const verifyEmail = asyncWrapper(async (req, res, next) => {
  await AuthService.verifyEmail(
    String(req.body.verificationToken),
    req.body.email,
  );
  res.status(200).json({
    status: HttpStatus.SUCCESS,
    message: "Email has been verified successfully.",
  });
});
///// verify email
module.exports = {
  register,
  login,
  forgotPassword,
  resetPassword,
  changePassword,
  verifyEmail,
};
