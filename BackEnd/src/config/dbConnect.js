const mongoose = require("mongoose");

const connectDB = async () => {
  const uri = process.env.DB_URI;
  const dbName = process.env.DB_NAME;

  if (!uri) {
    throw new Error("DB_URI is not configured");
  }

  if (!dbName) {
    throw new Error("DB_NAME is not configured");
  }

  try {
    await mongoose.connect(uri, {
      dbName,
      serverSelectionTimeoutMS: 5000,
      bufferCommands: false,
    });
    console.log(
      `Connected to MongoDB database "${mongoose.connection.name}" at ${mongoose.connection.host}`,
    );
  } catch (error) {
    console.error("Error connecting to MongoDB:", error.message);
    throw error;
  }
};

module.exports = connectDB;
