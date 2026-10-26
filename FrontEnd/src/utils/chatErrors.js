import { arabicMessage } from "./format.js";

export const CHAT_CONNECTION_ERROR =
  "تعذّر الاتصال بالخدمة الآن. تحقق من اتصالك ثم حاول مرة أخرى.";
export const CHAT_SERVER_ERROR =
  "تعذّر إكمال الطلب من الخادم. حاول مرة أخرى.";

export function chatRequestErrorMessage(requestError) {
  const response = requestError?.response;

  // A response means the service was reached.  Never blame the user's
  // connection for a server-side rejection such as a message limit or 5xx.
  if (!response) return CHAT_CONNECTION_ERROR;

  const serverMessage =
    response.data?.message || response.data?.error?.message;
  return arabicMessage(serverMessage, CHAT_SERVER_ERROR);
}
