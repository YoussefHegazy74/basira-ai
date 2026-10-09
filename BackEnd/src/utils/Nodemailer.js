const nodemailer = require("nodemailer");
async function sendEmail(options) {
  const transporter = nodemailer.createTransport({
    service: process.env.Service,
    auth: {
      user: process.env.USER,
      pass: process.env.APP_PASSWORD,
    },
    secure: true,
  });

  const info = await transporter.sendMail({
    from: "Basira AI",
    to: options.to,
    subject: options.subject,
    text: options.text,
    html: options.html,
  });
}

module.exports = sendEmail;
