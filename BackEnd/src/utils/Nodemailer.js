const nodemailer = require("nodemailer");

function createTransporter() {
  const service = process.env.SMTP_SERVICE;
  const user = process.env.SMTP_USER;
  const pass = process.env.SMTP_PASSWORD;

  if (!service || !user || !pass) {
    throw new Error(
      "SMTP is not configured: set SMTP_SERVICE, SMTP_USER and SMTP_PASSWORD",
    );
  }

  return nodemailer.createTransport({
    service,
    auth: { user, pass },
    secure: true,
  });
}

async function sendEmail(options) {
  const transporter = createTransporter();

  return await transporter.sendMail({
    from: process.env.SMTP_FROM || process.env.SMTP_USER,
    to: options.to,
    subject: options.subject,
    text: options.text,
    html: options.html,
  });
}

module.exports = sendEmail;
