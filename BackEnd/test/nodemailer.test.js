const { test } = require("node:test");
const assert = require("node:assert/strict");
const nodemailer = require("nodemailer");

const sendEmail = require("../src/utils/Nodemailer.js");

const originalCreateTransport = nodemailer.createTransport;

test("sendEmail authenticates with SMTP_USER instead of the OS USER", async (t) => {
  const transports = [];
  nodemailer.createTransport = (options) => {
    transports.push(options);
    return {
      sendMail: async (message) => ({ accepted: [message.to] }),
    };
  };

  const previous = {
    SMTP_SERVICE: process.env.SMTP_SERVICE,
    SMTP_USER: process.env.SMTP_USER,
    SMTP_PASSWORD: process.env.SMTP_PASSWORD,
    USER: process.env.USER,
  };
  process.env.USER = "os-user";
  process.env.SMTP_SERVICE = "gmail";
  process.env.SMTP_USER = "basira@example.com";
  process.env.SMTP_PASSWORD = "app-password";

  t.after(() => {
    nodemailer.createTransport = originalCreateTransport;
    for (const [key, value] of Object.entries(previous)) {
      if (value === undefined) {
        delete process.env[key];
      } else {
        process.env[key] = value;
      }
    }
  });

  await sendEmail({
    to: "user@example.com",
    subject: "Subject",
    text: "Text",
  });

  assert.equal(transports.length, 1);
  assert.equal(transports[0].service, "gmail");
  assert.deepEqual(transports[0].auth, {
    user: "basira@example.com",
    pass: "app-password",
  });
});

test("sendEmail fails fast when SMTP vars are missing", async (t) => {
  const previous = {
    SMTP_SERVICE: process.env.SMTP_SERVICE,
    SMTP_USER: process.env.SMTP_USER,
    SMTP_PASSWORD: process.env.SMTP_PASSWORD,
  };
  delete process.env.SMTP_SERVICE;
  delete process.env.SMTP_USER;
  delete process.env.SMTP_PASSWORD;

  t.after(() => {
    for (const [key, value] of Object.entries(previous)) {
      if (value !== undefined) {
        process.env[key] = value;
      }
    }
  });

  await assert.rejects(
    () => sendEmail({ to: "user@example.com", subject: "s", text: "t" }),
    /SMTP is not configured/,
  );
});
