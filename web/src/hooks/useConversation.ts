import { useCallback } from "react";
import { useWebSocket } from "./useWebSocket";
import { useAudioRecorder } from "./useAudioRecorder";
import { useAudioStore } from "../store/audioStore";

export function useConversation() {
  const isRecording = useAudioStore((s) => s.isRecording);
  const { sendAudio, sendStop } = useWebSocket();

  const onChunk = useCallback(
    (pcm: ArrayBuffer) => {
      sendAudio(pcm);
    },
    [sendAudio]
  );

  const { start, stop } = useAudioRecorder(onChunk);

  const toggleRecording = useCallback(async () => {
    if (isRecording) {
      stop();
      sendStop();
    } else {
      await start();
    }
  }, [isRecording, start, stop, sendStop]);

  return { isRecording, toggleRecording };
}
