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
    // Client-generated idempotency key for a user submission.  A retry after
    // a timeout or model failure must reuse this key so the backend can
    // recover the already-persisted user message instead of duplicating it.
    clientMessageId: {
      type: String,
      trim: true,
      maxlength: [128, "clientMessageId cannot exceed 128 characters"],
      default: null,
    },
    // Links an assistant message to the user message it answers.  Older
    // messages have no replyTo; the service falls back to chronological
    // adjacency for those.
    replyTo: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Message",
      default: null,
    },
  },
  { timestamps: { createdAt: true, updatedAt: false }, versionKey: false },
);

messageSchema.index({ conversationId: 1, createdAt: -1, _id: -1 });
messageSchema.index(
  { clientMessageId: 1 },
  {
    unique: true,
    partialFilterExpression: { clientMessageId: { $type: "string" } },
  },
);
messageSchema.index(
  { replyTo: 1 },
  {
    unique: true,
    partialFilterExpression: { replyTo: { $type: "objectId" } },
  },
);

const Message = mongoose.model("Message", messageSchema);
module.exports = Message;
