import { arabicMessage } from "./format.js";

export const LOGIN_CREDENTIALS_ERROR =
  "تعذّر تسجيل الدخول. راجع البريد الإلكتروني وكلمة المرور وحاول مرة أخرى.";
export const LOGIN_CONNECTION_ERROR =
  "تعذّر الاتصال بالخدمة الآن. تحقق من اتصالك ثم حاول مرة أخرى.";
export const LOGIN_SERVER_ERROR =
  "تعذّر إكمال تسجيل الدخول من الخادم. حاول لاحقًا.";
export const LOGIN_RATE_LIMIT_FALLBACK =
  "تم إرسال طلبات كثيرة. انتظر قليلًا ثم حاول مرة أخرى.";

export function loginErrorMessage(requestError) {
  const response = requestError?.response;
  if (!response) return LOGIN_CONNECTION_ERROR;

  const data = response.data;
  // express-rate-limit returns a plain string body, not { message }.
  const serverMessage =
    typeof data === "string"
      ? data
      : data?.message || data?.Message;

  if (response.status === 401) {
    return arabicMessage(serverMessage, LOGIN_CREDENTIALS_ERROR);
  }
  if (response.status === 429) {
    return arabicMessage(serverMessage, LOGIN_RATE_LIMIT_FALLBACK);
  }
  return arabicMessage(serverMessage, LOGIN_SERVER_ERROR);
}
