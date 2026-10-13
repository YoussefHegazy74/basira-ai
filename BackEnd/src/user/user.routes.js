const express = require("express");
const router = express.Router();
const userController = require("./user.controller.js");
const { protect } = require("../middleware/auth.middleware.js");
const restrictTo = require("../middleware/authorization.middleware.js");
router.use(protect);
router.use(restrictTo("admin"));
router.route("/").get(userController.getAllUsers);
router
  .route("/:id")
  .get(userController.getUserById)
  .patch(userController.updateUserById)
  .delete(userController.deleteUser);

module.exports = router;
