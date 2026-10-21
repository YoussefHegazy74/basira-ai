
import api from "./api";

export async function getTafsir(ayah) {
  const response = await api.post("/tafsir", {
    ayah: ayah,
  });

  return response.data;
}

export async function verifyText(text, explanation) {
  const response = await api.post("/verify", {
    text: text,
    explanation: explanation,
  });

  return response.data;
}
