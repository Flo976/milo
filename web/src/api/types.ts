// STT
export interface STTRequest {
  audio: string;
  format?: string;
  sample_rate?: number;
}

export interface STTResponse {
  text: string;
  language: string;
  confidence: number;
  duration_ms: number;
  processing_ms: number;
}

// TTS
export interface TTSRequest {
  text: string;
  language?: string;
  format?: string;
}

export interface TTSResponse {
  audio: string;
  format: string;
  sample_rate: number;
  processing_ms: number;
}

// Chat
export interface ChatRequest {
  message: string;
  session_id?: string;
  language?: string;
  mode?: string;
}

export interface ChatResponse {
  reply: string;
  session_id: string;
  mode: string;
  processing_ms: number;
}

// Translate
export interface TranslateRequest {
  text: string;
  source?: string;
  target?: string;
}

export interface TranslateResponse {
  translation: string;
  source: string;
  target: string;
  processing_ms: number;
}

// Health
export interface HealthResponse {
  status: string;
  version: string;
  mode: string;
  models_loaded: string[];
  vram: {
    available: boolean;
    allocated_gb?: number;
    reserved_gb?: number;
    total_gb?: number;
    free_gb?: number;
  };
  redis_connected: boolean;
}

// WebSocket events
export type WSEventType =
  | "vad"
  | "transcript"
  | "reply_text"
  | "reply_audio"
  | "mode"
  | "session"
  | "error";

export interface WSEvent {
  type: WSEventType;
  speaking?: boolean;
  text?: string;
  partial?: boolean;
  audio?: string;
  value?: string;
  session_id?: string;
  error?: string;
}

// Conversation
export interface Message {
  id: string;
  role: "user" | "assistant";
  text: string;
  audio?: string;
  timestamp: number;
}
