import { create } from "zustand";

interface AudioState {
  isRecording: boolean;
  isPlaying: boolean;
  micPermission: "prompt" | "granted" | "denied";

  setRecording: (v: boolean) => void;
  setPlaying: (v: boolean) => void;
  setMicPermission: (v: "prompt" | "granted" | "denied") => void;
}

export const useAudioStore = create<AudioState>((set) => ({
  isRecording: false,
  isPlaying: false,
  micPermission: "prompt",

  setRecording: (v) => set({ isRecording: v }),
  setPlaying: (v) => set({ isPlaying: v }),
  setMicPermission: (v) => set({ micPermission: v }),
}));
