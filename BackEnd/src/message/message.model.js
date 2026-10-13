const mongoose = require("mongoose");

const messageSchema = new mongoose.Schema(
  {
    conversationId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Conversation",
      required: true,
    },
    role: {
      type: String,
      enum: ["user", "assistant"],
      required: true,
    },
    sources: { type: [String], default: [] },
    content: {
      type: String,
      required: true,
      trim: true,
      maxlength: [20000, "Content cannot exceed 20,000 characters"],
    },
  },
  { timestamps: { createdAt: true, updatedAt: false }, versionKey: false },
);

messageSchema.index({ conversationId: 1, createdAt: -1 });

const Message = mongoose.model("Message", messageSchema);
module.exports = Message;
