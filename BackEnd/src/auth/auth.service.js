const User = require("../user/user.model.js");
const bcrypt = require("bcrypt");
const jwt = require("jsonwebtoken");
const crypto = require("crypto");
const AppError = require("../utils/AppError.js");
const HttpStatus = require("../utils/HttpStatusText.js");
const sendEmail = require("../utils/Nodemailer.js");
const pendingUserModel = require("../user/pendingusers.model.js");
const VERIFICATION_CODE_TTL_MS = 5 * 60 * 1000;
const validator = require("validator");
const generateVerificationCode = () =>
  crypto.randomInt(0, 100_000_000).toString().padStart(8, "0");

const signToken = (id) =>
  jwt.sign({ id }, process.env.JWT_SECRET, {
    expiresIn: process.env.JWT_EXPIRES_IN || "7d",
  });

const buildVerificationEmail = (code) => ({
  subject: "Email Verification",
  text: `Your verification code is: ${code}. It expires in 5 minutes.`,
  html: `
    <div style="font-family: Arial, sans-serif; max-width: 500px; margin: auto; padding: 24px; text-align: center; border: 1px solid #e5e7eb; border-radius: 12px;">
      <h2 style="color: #111827;">Email Verification</h2>
      <p style="color: #4b5563;">Use the code below to verify your email address.</p>
      <div style="margin: 24px 0; padding: 16px; background-color: #f3f4f6; border-radius: 8px;">
        <h1 style="letter-spacing: 8px; color: #2563eb; margin: 0;">${code}</h1>
      </div>
      <p style="color: #6b7280; font-size: 14px;">This code expires in 5 minutes.</p>
      <p style="color: #9ca3af; font-size: 12px;">If you didn't request this code, you can ignore this email.</p>
    </div>
  `,
});

const register = async ({
  firstName,
  lastName,
  email,
  password,
  passwordconfirm,
}) => {
  if (await User.exists({ email })) {
    throw new AppError(409, "Email already exists", HttpStatus.FAIL);
  }
  await pendingUserModel.deleteMany({ email });
  const verificationToken = generateVerificationCode();
  const tempUser = await pendingUserModel.create({
    firstName,
    lastName,
    email,
    password,
    passwordconfirm,
    VerificationToken: verificationToken,
    verificationTokenExpiresAt: new Date(Date.now() + VERIFICATION_CODE_TTL_MS),
  });
  try {
    await sendEmail({
      to: tempUser.email,
      ...buildVerificationEmail(verificationToken),
    });
  } catch (error) {
    console.error("Email sending failed:", error.message);
    await pendingUserModel.deleteOne({ _id: tempUser._id }).catch(() => {});

    throw new AppError(
      500,
      "Failed to send verification email",
      HttpStatus.ERROR,
    );
  }
  return;
};
//// log in
const login = async (userData) => {
  const email = String(userData.email || "")
    .trim()
    .toLowerCase();
  const user = await User.findOne({ email }).select("+password");

  if (!user || !(await user.correctPassword(userData.password))) {
    throw new AppError(401, "Incorrect email or password", HttpStatus.FAIL);
  }

  // Deactivation must not be reversible by logging in again.  Only an admin
  // reactivation flow may set isActive back to true.
  if (!user.isActive) {
    throw new AppError(403, "Account is deactivated", HttpStatus.FAIL);
  }

  const token = signToken(user._id);

  return {
    token,
    user: {
      id: String(user._id),
      firstName: user.firstName,
      lastName: user.lastName,
      email: user.email,
      role: user.role === "admin" ? "admin" : "user",
    },
  };
};

const forgotPassword = async (req, res) => {
  ///1 check if user exist

  const user = await User.findOne({ email: req.body.email });
  // Do not reveal whether an account exists.  The controller always returns
  // the same generic message for known and unknown addresses.
  if (!user) return;
  const resetToken = user.createPasswordResetToken();
  await user.save({ validateBeforeSave: false });
  const resetURL = `${process.env.FRONTEND_URL}/auth/resetpassword/${resetToken}`;

  await sendEmail({
    to: user.email,
    subject: "Password Reset",

    text: `You requested a password reset. Click the following link to reset your password: ${resetURL}`,

    html: `
      <div style="
        font-family: Arial, Helvetica, sans-serif;
        background-color: #f4f7fb;
        padding: 40px 15px;
        color: #333333;
      ">
        <div style="
          max-width: 500px;
          margin: 0 auto;
          background-color: #ffffff;
          border-radius: 12px;
          overflow: hidden;
          box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        ">

          <!-- Header -->
          <div style="
            background-color: #2563eb;
            padding: 25px;
            text-align: center;
          ">
            <h1 style="
              color: #ffffff;
              font-size: 24px;
              margin: 0;
            ">
              Password Reset
            </h1>
          </div>

          <!-- Email Content -->
          <div style="padding: 30px;">

            <h2 style="
              color: #1f2937;
              font-size: 20px;
              margin-top: 0;
            ">
              Hello,
            </h2>

            <p style="
              font-size: 15px;
              line-height: 1.7;
              color: #4b5563;
            ">
              We received a request to reset your password.
              Click the button below to choose a new password.
            </p>

            <!-- Reset Button -->
            <div style="text-align: center; margin: 30px 0;">
              <a href="${resetURL}" style="
                display: inline-block;
                background-color: #2563eb;
                color: #ffffff;
                text-decoration: none;
                padding: 14px 30px;
                border-radius: 6px;
                font-size: 15px;
                font-weight: bold;
              ">
                Reset Password
              </a>
            </div>

            <p style="
              font-size: 13px;
              line-height: 1.6;
              color: #6b7280;
            ">
              If you didn't request a password reset, you can
              safely ignore this email. Your password will remain unchanged.
            </p>

          </div>

          <!-- Footer -->
          <div style="
            background-color: #f9fafb;
            padding: 18px;
            text-align: center;
          ">
            <p style="
              color: #9ca3af;
              font-size: 12px;
              margin: 0;
            ">
              &copy; ${new Date().getFullYear()} Basira AI.
              All rights reserved.
            </p>
          </div>

        </div>
      </div>
    `,
  });
  return;
};
//// reset password
const resetPassword = async (token, newPassword, newPasswordConfirm) => {
  const hashedToken = crypto.createHash("sha256").update(token).digest("hex");
  const user = await User.findOne({
    resetPasswordToken: hashedToken,
    resetPasswordTokenExpiresAt: { $gt: Date.now() },
  });
  if (!user) {
    throw new AppError(
      400,
      "Token is invalid or has expired",
      HttpStatus.BAD_REQUEST,
    );
  }
  if (newPassword !== newPasswordConfirm) {
    throw new AppError(400, "Passwords do not match", HttpStatus.BAD_REQUEST);
  }
  if (
    !validator.isStrongPassword(newPassword, {
      minLength: 8,
      minLowercase: 1,
      minUppercase: 1,
      minNumbers: 1,
      minSymbols: 1,
    })
  ) {
    throw new AppError(
      400,
      "Password must be at least 8 characters and include uppercase, lowercase, number and symbol (@, $, ...)",
      HttpStatus.BAD_REQUEST,
    );
  }
  user.password = await bcrypt.hash(newPassword, 10);
  user.resetPasswordToken = undefined;
  user.resetPasswordTokenExpiresAt = undefined;
  user.passwordChangedAt = Date.now() - 1000;
  await user.save();
};
/// change password
const changePassword = async (userId, currentPassword, newPassword) => {
  const user = await User.findById(userId).select("+password");
  if (!user || !(await user.correctPassword(currentPassword))) {
    throw new AppError(401, "Incorrect current password", HttpStatus.FAIL);
  }
  if (
    !validator.isStrongPassword(newPassword, {
      minLength: 8,
      minLowercase: 1,
      minUppercase: 1,
      minNumbers: 1,
      minSymbols: 1,
    })
  ) {
    throw new AppError(
      400,
      "Password must be at least 8 characters and include uppercase, lowercase, number and symbol (@, $, ...)",
      HttpStatus.BAD_REQUEST,
    );
  }
  user.password = await bcrypt.hash(newPassword, 10);
  user.passwordChangedAt = Date.now() - 1000;
  await user.save();
  // new token, because the old one is invalidated by passwordChangedAt
  return signToken(user._id);
};

/// verify email
const verifyEmail = async (token, email) => {
  if (typeof email !== "string" || typeof token !== "string") {
    throw new AppError(400, "Email and code are required", HttpStatus.FAIL);
  }
  const pendingUser = await pendingUserModel
    .findOne({
      email: email,
    })
    .select("+password");
  if (!pendingUser) {
    throw new AppError(400, "Email not found", HttpStatus.FAIL);
  }
  if (pendingUser.verificationTokenExpiresAt < Date.now()) {
    await pendingUserModel.deleteOne({ _id: pendingUser._id });
    throw new AppError(400, "Verification token has expired", HttpStatus.FAIL);
  }
  if (pendingUser.VerificationToken !== token) {
    pendingUser.limit -= 1;
    await pendingUser.save();
    if (pendingUser.limit <= 0) {
      await pendingUserModel.deleteOne({ _id: pendingUser._id });
      throw new AppError(
        400,
        "Verification token has expired",
        HttpStatus.FAIL,
      );
    }
    throw new AppError(400, "Invalid verification token", HttpStatus.FAIL);
  }
  await User.create({
    firstName: pendingUser.firstName,
    lastName: pendingUser.lastName,
    email: pendingUser.email,
    password: pendingUser.password,
  });
  await pendingUserModel.deleteOne({ _id: pendingUser._id });
  return;
};
///// logout
const logout = async (req, res) => {
  res.clearCookie("token", {
    httpOnly: true,
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
  });
  return;
};
///// deactivate account
const deactivateAccount = async (userId) => {
  const user = await User.findById(userId);

  if (!user) {
    throw new AppError(404, "User not found", HttpStatus.FAIL);
  }

  if (!user.isActive) {
    throw new AppError(400, "Account is already deactivated", HttpStatus.FAIL);
  }

  if (user.role === "admin") {
    const otherAdmins = await User.countDocuments({
      role: "admin",
      _id: { $ne: user._id },
    });
    if (otherAdmins === 0) {
      throw new AppError(
        400,
        "Cannot deactivate the last administrator",
        HttpStatus.BAD_REQUEST,
      );
    }
  }

  user.isActive = false;
  user.expiresAt = new Date(Date.now() + 30 * 24 * 60 * 60 * 1000);

  await user.save();
  return;
};
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
