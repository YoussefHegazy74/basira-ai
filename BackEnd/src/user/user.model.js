const mongoose = require("mongoose");
const bcrypt = require("bcrypt");
const { randomBytes, createHash } = require("crypto");
const validator = require("validator");

const userSchema = new mongoose.Schema(
  {
    firstName: {
      type: String,
      required: [true, "الاسم الأول مطلوب"],
      trim: true,
    },

    lastName: {
      type: String,
      required: [true, "اسم العائلة مطلوب"],
      trim: true,
    },

    email: {
      type: String,
      required: [true, "البريد الإلكتروني مطلوب"],
      unique: [true, "البريد الإلكتروني مستخدم بالفعل"],
      lowercase: true,
      trim: true,
      validate: [validator.isEmail, "يرجى إدخال بريد إلكتروني صالح"],
    },
    password: {
      type: String,
      required: [true, "كلمة المرور مطلوبة"],
      select: false,
    },

    passwordChangedAt: {
      type: Date,
      default: null,
    },

    role: {
      type: String,
      enum: ["user", "admin"],
      default: "user",
    },

    resetPasswordToken: {
      type: String,
      default: null,
      select: false,
    },

    resetPasswordTokenExpiresAt: {
      type: Date,
      default: null,
      select: false,
    },
    expiresAt: {
      type: Date,
      expires: 0,
    },
    isActive: {
      type: Boolean,
      default: true,
    },
  },
  {
    timestamps: true,
    versionKey: false,
  },
);

userSchema.methods.correctPassword = async function (candidatePassword) {
  return await bcrypt.compare(candidatePassword, this.password);
};

userSchema.methods.changedPasswordAfter = function (JWTTimestamp) {
  if (this.passwordChangedAt) {
    const changedTimestamp = parseInt(
      this.passwordChangedAt.getTime() / 1000,
      10,
    );
    return JWTTimestamp < changedTimestamp;
  }
  return false;
};

userSchema.methods.createPasswordResetToken = function () {
  const resetToken = randomBytes(32).toString("hex");
  this.resetPasswordToken = createHash("sha256")
    .update(resetToken)
    .digest("hex");
  this.resetPasswordTokenExpiresAt = Date.now() + 5 * 60 * 1000;
  return resetToken;
};

module.exports = mongoose.model("User", userSchema);
