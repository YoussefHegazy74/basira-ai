const mongoose = require("mongoose");
const validator = require("validator");
const bcrypt = require("bcrypt");
const jwt = require("jsonwebtoken");
const pendingUserSchema = new mongoose.Schema(
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
      unique: [true, "البريد الإلكتروني مستخدم بالفعل"],
      required: [true, "البريد الإلكتروني مطلوب"],
      lowercase: true,
      trim: true,
      validate: [validator.isEmail, "يرجى إدخال بريد إلكتروني صالح"],
    },
    password: {
      type: String,
      required: [true, "كلمة المرور مطلوبة"],
      validate: {
        validator: (value) =>
          validator.isStrongPassword(value, {
            minLength: 8,
            minLowercase: 1,
            minUppercase: 1,
            minNumbers: 1,
            minSymbols: 1,
          }),
        message:
          "يجب أن تكون كلمة المرور 8 أحرف على الأقل وتتضمن حرفًا كبيرًا وحرفًا صغيرًا ورقمًا ورمزًا (@، $، ...)",
      },
      select: false,
    },
    passwordconfirm: {
      type: String,
      select: false,
      required: [true, "تأكيد كلمة المرور مطلوب"],
      validate: {
        validator: function (el) {
          return el === this.password;
        },
        message: "كلمتا المرور غير متطابقتين",
      },
    },
    VerificationToken: {
      type: String,
      default: null,
    },
    verificationTokenExpiresAt: {
      type: Date,
      default: null,
      index: { expireAfterSeconds: 0 },
    },
    limit: {
      type: Number,
      default: 5,
    },
  },
  { timestamps: true, versionKey: false },
);

pendingUserSchema.pre("save", async function () {
  if (!this.isModified("password")) return;
  this.password = await bcrypt.hash(this.password, 10);
  this.passwordconfirm = undefined;
});

const PendingUser = mongoose.model("PendingUser", pendingUserSchema);
module.exports = PendingUser;
