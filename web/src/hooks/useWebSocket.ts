import { useEffect, useRef } from "react";
import { wsManager } from "../api/ws";
import { useConversationStore } from "../store/conversationStore";
import { useSettingsStore } from "../store/settingsStore";
import { AudioPlayer } from "../audio/player";
import type { WSEvent } from "../api/types";

const player = new AudioPlayer();

export function useWebSocket() {
  const apiKey = useSettingsStore((s) => s.apiKey);
  const autoPlayRef = useRef(useSettingsStore.getState().autoPlay);

  // Keep ref in sync without triggering effect re-runs
  useEffect(() => {
    return useSettingsStore.subscribe((s) => {
      autoPlayRef.current = s.autoPlay;
    });
  }, []);

  // Store actions are stable refs in Zustand — grab once
  const storeActions = useRef({
    addMessage: useConversationStore.getState().addMessage,
    setSessionId: useConversationStore.getState().setSessionId,
    setMode: useConversationStore.getState().setMode,
    setSpeaking: useConversationStore.getState().setSpeaking,
    setLiveTranscript: useConversationStore.getState().setLiveTranscript,
  });

  useEffect(() => {
    if (!apiKey) return;

    const { addMessage, setSessionId, setMode, setSpeaking, setLiveTranscript } =
      storeActions.current;

    const handleEvent = (evt: WSEvent) => {
      switch (evt.type) {
        case "session":
          if (evt.session_id) setSessionId(evt.session_id);
          break;
        case "vad":
          setSpeaking(evt.speaking ?? false);
          break;
        case "transcript":
          setLiveTranscript(evt.text ?? "");
          if (!evt.partial) {
            addMessage({
              id: "",
              role: "user",
              text: evt.text ?? "",
              timestamp: Date.now(),
            });
            setLiveTranscript("");
          }
          break;
        case "reply_text":
          addMessage({
            id: "",
            role: "assistant",
            text: evt.text ?? "",
            timestamp: Date.now(),
          });
          break;
        case "reply_audio":
          if (autoPlayRef.current && evt.audio) {
            void player.playBase64(evt.audio);
          }
          break;
        case "mode":
          if (evt.value === "local" || evt.value === "cloud" || evt.value === "echo") {
            setMode(evt.value);
          }
          break;
      }
    };

    const unsub = wsManager.onEvent(handleEvent);
    wsManager.connect(apiKey);

    return () => {
      unsub();
      wsManager.disconnect();
    };
  }, [apiKey]); // Only re-run when apiKey changes

  return {
    sendAudio: (pcm: ArrayBuffer) => wsManager.sendAudio(pcm),
    sendStop: () => wsManager.sendControl({ type: "stop" }),
    connected: wsManager.connected,
  };
}
