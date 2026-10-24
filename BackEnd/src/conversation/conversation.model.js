const mongoose = require("mongoose");

const conversationSchema = new mongoose.Schema(
  {
    userId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      required: true,
    },
    title: { type: String, default: "محادثة جديدة", trim: true, maxlength: 100 },
    messageCount: { type: Number, default: 0, min: 0 },
  },
  { timestamps: true, versionKey: false },
);
conversationSchema.index({ userId: 1, updatedAt: -1 });

const Conversation = mongoose.model("Conversation", conversationSchema);
module.exports = Conversation;
