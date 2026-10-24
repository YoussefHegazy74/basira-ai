const userService = require("./user.service.js");
const User = require("./user.model.js");
const AppError = require("../utils/AppError.js");
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
  const isSelf = String(req.user._id) === String(req.params.id);
  const isAdmin = req.user.role === "admin";

  if (!isSelf && !isAdmin) {
    throw new AppError(
      403,
      "You are not allowed to update this user",
      HttpStatus.FAIL,
    );
  }

  // Never pass req.body directly to the model: a normal user must not be
  // able to change their role, password, reset tokens, or account status.
  const allowed = ["firstName", "lastName", "email"];
  if (isAdmin) allowed.push("role", "isActive");
  const updates = {};
  for (const field of allowed) {
    if (Object.prototype.hasOwnProperty.call(req.body, field)) {
      updates[field] = req.body[field];
    }
  }
  if (Object.prototype.hasOwnProperty.call(updates, "isActive")) {
    const active = updates.isActive === true || updates.isActive === "true";
    if (!active) {
      const target = await User.findById(req.params.id).select("role");
      if (target?.role === "admin") {
        const otherAdmins = await User.countDocuments({
          role: "admin",
          _id: { $ne: target._id },
        });
        if (otherAdmins === 0) {
          throw new AppError(
            400,
            "Cannot deactivate the last administrator",
            HttpStatus.BAD_REQUEST,
          );
        }
      }
    }
    updates.isActive = active;
    // The TTL field is what actually removes a deactivated account.  Clear
    // it on reactivation and restore the 30-day window on deactivation.
    updates.expiresAt = active
      ? null
      : new Date(Date.now() + 30 * 24 * 60 * 60 * 1000);
  }
  if (Object.keys(updates).length === 0) {
    throw new AppError(
      400,
      "No supported fields to update",
      HttpStatus.BAD_REQUEST,
    );
  }

  const user = await userService.updateUserById(req.params.id, updates);
  res.status(200).json({ status: HttpStatus.SUCCESS, data: user });
});

// delete user
const deleteUser = asyncWrapper(async (req, res) => {
  if (String(req.user._id) === String(req.params.id)) {
    throw new AppError(
      400,
      "You cannot delete your own account",
      HttpStatus.BAD_REQUEST,
    );
  }

  const target = await User.findById(req.params.id).select("role");
  if (!target) {
    throw new AppError(404, "User not found", HttpStatus.FAIL);
  }
  if (target.role === "admin") {
    const otherAdmins = await User.countDocuments({
      role: "admin",
      _id: { $ne: target._id },
    });
    if (otherAdmins === 0) {
      throw new AppError(
        400,
        "Cannot delete the last administrator",
        HttpStatus.BAD_REQUEST,
      );
    }
  }

  const user = await userService.deleteUser(req.params.id);
  res.status(200).json({ status: HttpStatus.SUCCESS, data: user });
});

module.exports = {
  getAllUsers,
  getUserById,
  updateUserById,
  deleteUser,
};
