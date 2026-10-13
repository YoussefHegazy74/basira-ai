const userService = require("./user.service.js");
const HttpStatus = require("../utils/HttpStatusText.js");
const asyncWrapper = require("../middleware/Async_wrapper.js");

// get all users
const getAllUsers = asyncWrapper(async (req, res) => {
  const users = await userService.getAllUsers();
  res.status(200).json({ status: HttpStatus.SUCCESS, data: users });
});

// get user by id
const getUserById = asyncWrapper(async (req, res) => {
  const user = await userService.getUserById(req.params.id);
  res.status(200).json({ status: HttpStatus.SUCCESS, data: user });
});

// update user by id
const updateUserById = asyncWrapper(async (req, res) => {
  const user = await userService.updateUserById(req.params.id, req.body);
  res.status(200).json({ status: HttpStatus.SUCCESS, data: user });
});

// delete user
const deleteUser = asyncWrapper(async (req, res) => {
  const user = await userService.deleteUser(req.params.id);
  res.status(200).json({ status: HttpStatus.SUCCESS, data: user });
});

module.exports = {
  getAllUsers,
  getUserById,
  updateUserById,
  deleteUser,
};
