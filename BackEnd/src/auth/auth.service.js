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
  subject: "التحقق من البريد الإلكتروني",
  text: `رمز التحقق الخاص بك هو: ${code}. تنتهي صلاحيته خلال 5 دقائق.`,
  html: `
    <div style="font-family: Arial, sans-serif; max-width: 500px; margin: auto; padding: 24px; text-align: center; border: 1px solid #e5e7eb; border-radius: 12px;">
      <h2 style="color: #111827;">التحقق من البريد الإلكتروني</h2>
      <p style="color: #4b5563;">استخدم الرمز أدناه للتحقق من عنوان بريدك الإلكتروني.</p>
      <div style="margin: 24px 0; padding: 16px; background-color: #f3f4f6; border-radius: 8px;">
        <h1 style="letter-spacing: 8px; color: #2563eb; margin: 0;">${code}</h1>
      </div>
      <p style="color: #6b7280; font-size: 14px;">تنتهي صلاحية هذا الرمز خلال 5 دقائق.</p>
      <p style="color: #9ca3af; font-size: 12px;">إذا لم تطلب هذا الرمز، يمكنك تجاهل هذا البريد الإلكتروني.</p>
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
    throw new AppError(409, "البريد الإلكتروني مستخدم بالفعل", HttpStatus.FAIL);
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
      "فشل إرسال بريد التحقق",
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
    throw new AppError(401, "البريد الإلكتروني أو كلمة المرور غير صحيحة", HttpStatus.FAIL);
  }

  const token = signToken(user._id);

  user.isActive = true;
  user.expiresAt = undefined;

  await user.save({ validateBeforeSave: false });

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
  if (!user) {
    throw new AppError(404, "المستخدم غير موجود", HttpStatus.FAIL);
  }
  const resetToken = user.createPasswordResetToken();
  await user.save({ validateBeforeSave: false });
  const resetURL = `${process.env.FRONTEND_URL}/auth/resetpassword/${resetToken}`;

  await sendEmail({
    to: user.email,
    subject: "إعادة تعيين كلمة المرور",

    text: `لقد طلبت إعادة تعيين كلمة المرور. انقر على الرابط التالي لإعادة تعيين كلمة المرور: ${resetURL}`,

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
              إعادة تعيين كلمة المرور
            </h1>
          </div>

          <!-- Email Content -->
          <div style="padding: 30px;">

            <h2 style="
              color: #1f2937;
              font-size: 20px;
              margin-top: 0;
            ">
              مرحبًا،
            </h2>

            <p style="
              font-size: 15px;
              line-height: 1.7;
              color: #4b5563;
            ">
              تلقينا طلبًا لإعادة تعيين كلمة المرور الخاصة بك.
              انقر على الزر أدناه لاختيار كلمة مرور جديدة.
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
                إعادة تعيين كلمة المرور
              </a>
            </div>

            <p style="
              font-size: 13px;
              line-height: 1.6;
              color: #6b7280;
            ">
              إذا لم تطلب إعادة تعيين كلمة المرور، يمكنك
              تجاهل هذا البريد الإلكتروني بأمان. ستبقى كلمة المرور الخاصة بك دون تغيير.
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
              جميع الحقوق محفوظة.
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
      "الرمز غير صالح أو منتهي الصلاحية",
      HttpStatus.BAD_REQUEST,
    );
  }
  if (newPassword !== newPasswordConfirm) {
    throw new AppError(400, "كلمتا المرور غير متطابقتين", HttpStatus.BAD_REQUEST);
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
      "يجب أن تكون كلمة المرور 8 أحرف على الأقل وتتضمن حرفًا كبيرًا وحرفًا صغيرًا ورقمًا ورمزًا (@، $، ...)",
      HttpStatus.BAD_REQUEST,
    );
  }
  user.password = await bcrypt.hash(newPassword, 10);
  user.resetPasswordToken = undefined;
  user.resetPasswordTokenExpiresAt = undefined;
  user.passwordChangedAt = Date.now() - 1000;
  user.isActive = true;
  user.expiresAt = undefined;
  await user.save();
};
/// change password
const changePassword = async (userId, currentPassword, newPassword) => {
  const user = await User.findById(userId).select("+password");
  if (!user || !(await user.correctPassword(currentPassword))) {
    throw new AppError(401, "كلمة المرور الحالية غير صحيحة", HttpStatus.FAIL);
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
      "يجب أن تكون كلمة المرور 8 أحرف على الأقل وتتضمن حرفًا كبيرًا وحرفًا صغيرًا ورقمًا ورمزًا (@، $، ...)",
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
    throw new AppError(400, "البريد الإلكتروني والرمز مطلوبان", HttpStatus.FAIL);
  }
  const pendingUser = await pendingUserModel
    .findOne({
      email: email,
    })
    .select("+password");
  if (!pendingUser) {
    throw new AppError(400, "البريد الإلكتروني غير موجود", HttpStatus.FAIL);
  }
  if (pendingUser.verificationTokenExpiresAt < Date.now()) {
    await pendingUserModel.deleteOne({ _id: pendingUser._id });
    throw new AppError(400, "انتهت صلاحية رمز التحقق", HttpStatus.FAIL);
  }
  if (pendingUser.VerificationToken !== token) {
    pendingUser.limit -= 1;
    await pendingUser.save();
    if (pendingUser.limit <= 0) {
      await pendingUserModel.deleteOne({ _id: pendingUser._id });
      throw new AppError(
        400,
        "انتهت صلاحية رمز التحقق",
        HttpStatus.FAIL,
      );
    }
    throw new AppError(400, "رمز التحقق غير صالح", HttpStatus.FAIL);
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
    secure: true,
  });
  return;
};
///// deactivate account
const deactivateAccount = async (userId) => {
  const user = await User.findById(userId);

  if (!user) {
    throw new AppError(404, "المستخدم غير موجود", HttpStatus.FAIL);
  }

  if (!user.isActive) {
    throw new AppError(400, "الحساب معطّل بالفعل", HttpStatus.FAIL);
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
