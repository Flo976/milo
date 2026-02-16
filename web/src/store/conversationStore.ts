import { create } from "zustand";
import type { Message } from "../api/types";

interface ConversationState {
  messages: Message[];
  sessionId: string;
  mode: "local" | "cloud" | "echo" | "";
  isSpeaking: boolean;
  liveTranscript: string;
  isProcessing: boolean;

  addMessage: (msg: Message) => void;
  setSessionId: (id: string) => void;
  setMode: (mode: "local" | "cloud" | "echo") => void;
  setSpeaking: (v: boolean) => void;
  setLiveTranscript: (t: string) => void;
  setProcessing: (v: boolean) => void;
  clearMessages: () => void;
}

let msgCounter = 0;

export const useConversationStore = create<ConversationState>((set) => ({
  messages: [],
  sessionId: "",
  mode: "",
  isSpeaking: false,
  liveTranscript: "",
  isProcessing: false,

  addMessage: (msg) =>
    set((s) => ({
      messages: [...s.messages, { ...msg, id: msg.id || String(++msgCounter) }],
    })),

  setSessionId: (id) => set({ sessionId: id }),
  setMode: (mode) => set({ mode }),
  setSpeaking: (v) => set({ isSpeaking: v }),
  setLiveTranscript: (t) => set({ liveTranscript: t }),
  setProcessing: (v) => set({ isProcessing: v }),
  clearMessages: () => set({ messages: [], liveTranscript: "" }),
}));
