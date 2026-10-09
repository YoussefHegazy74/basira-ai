class appError extends Error {
  constructor(statusCode, message, statustext) {
    super(message);
    this.statusCode = statusCode;
    this.statustext = statustext;
  }
}
module.exports = appError;
