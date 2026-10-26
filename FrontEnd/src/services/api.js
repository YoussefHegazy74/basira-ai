
import axios from "axios";
import { alignLoopbackApiUrl } from "../utils/apiUrl";

const configuredBaseURL =
  import.meta.env.VITE_API_URL || "http://localhost:3000";
const baseURL = alignLoopbackApiUrl(
  configuredBaseURL,
  typeof window === "undefined" ? "" : window.location.hostname,
);

const api = axios.create({
  baseURL,
  withCredentials: true,
  headers: {
    "Content-Type": "application/json",
  },
});

export default api;