import { apiClient } from "./client";
import type {
  STTRequest,
  STTResponse,
  TTSRequest,
  TTSResponse,
  ChatRequest,
  ChatResponse,
  TranslateRequest,
  TranslateResponse,
  HealthResponse,
} from "./types";

export const api = {
  health: () => apiClient.fetch<HealthResponse>("/api/v1/health"),

  stt: (req: STTRequest) =>
    apiClient.fetch<STTResponse>("/api/v1/stt", {
      method: "POST",
      body: JSON.stringify(req),
    }),

  tts: (req: TTSRequest) =>
    apiClient.fetchRaw("/api/v1/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...req, format: "wav" }),
    }),

  ttsBase64: (req: TTSRequest) =>
    apiClient.fetch<TTSResponse>("/api/v1/tts", {
      method: "POST",
      body: JSON.stringify({ ...req, format: "base64" }),
    }),

  chat: (req: ChatRequest) =>
    apiClient.fetch<ChatResponse>("/api/v1/chat", {
      method: "POST",
      body: JSON.stringify(req),
    }),

  translate: (req: TranslateRequest) =>
    apiClient.fetch<TranslateResponse>("/api/v1/translate", {
      method: "POST",
      body: JSON.stringify(req),
    }),
};
