const app = require("./app");
const dotenv = require("dotenv");
const path = require("path");
const connectDB = require("./config/dbConnect");
dotenv.config({
  path: path.resolve(__dirname, "../../.env"),
});
connectDB();
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
