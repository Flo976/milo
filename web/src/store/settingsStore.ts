import { create } from "zustand";
import { persist } from "zustand/middleware";

interface SettingsState {
  apiKey: string;
  language: "mg" | "fr";
  autoPlay: boolean;

  setApiKey: (key: string) => void;
  setLanguage: (lang: "mg" | "fr") => void;
  setAutoPlay: (v: boolean) => void;
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      apiKey: "",
      language: "mg",
      autoPlay: true,

      setApiKey: (key) => set({ apiKey: key }),
      setLanguage: (lang) => set({ language: lang }),
      setAutoPlay: (v) => set({ autoPlay: v }),
    }),
    { name: "milo-settings" }
  )
);
