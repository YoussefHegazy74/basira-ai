const mongoose = require("mongoose");

const usageSchema = new mongoose.Schema(
  {
    ownerKey: { type: String, required: true },
    count: { type: Number, default: 0 },
    expiresAt: {
      type: Date,
      required: true,
      default: () => {
        const tomorrow = new Date();
        tomorrow.setDate(tomorrow.getDate() + 1);
        tomorrow.setHours(0, 0, 0, 0);
        return tomorrow;
      },
    },
  },
  { timestamps: true, versionKey: false },
);

usageSchema.index({ ownerKey: 1 }, { unique: true });
usageSchema.index({ expiresAt: 1 }, { expireAfterSeconds: 0 });

module.exports = mongoose.model("Usage", usageSchema);
