const User = require("./user.model.js");
const AppError = require("../utils/AppError.js");
const HttpStatus = require("../utils/HttpStatusText.js");

//// get all users
const getAllUsers = async () => {
  const users = await User.find();
  return users;
};

/// get user by id
const getUserById = async (userId) => {
  const user = await User.findById(userId);
  if (!user) {
    throw new AppError(404, "User not found", HttpStatus.FAIL);
  }
  return user;
};

/// update user by id
const updateUserById = async (userId, updateData) => {
  const user = await User.findByIdAndUpdate(userId, updateData, {
    returnDocument: "after",
    runValidators: true,
  });
  if (!user) {
    throw new AppError(404, "User not found", HttpStatus.FAIL);
  }
  return user;
};

//// delete user
const deleteUser = async (userId) => {
  const user = await User.findByIdAndDelete(userId);
  if (!user) {
    throw new AppError(404, "User not found", HttpStatus.FAIL);
  }
  return user;
};

module.exports = {
  getAllUsers,
  getUserById,
  updateUserById,
  deleteUser,
};
