import { useCallback, useRef } from "react";
import { AudioRecorder } from "../audio/recorder";
import { useAudioStore } from "../store/audioStore";

export function useAudioRecorder(onChunk: (pcm: ArrayBuffer) => void) {
  const recorder = useRef(new AudioRecorder());
  const setRecording = useAudioStore((s) => s.setRecording);
  const setMicPermission = useAudioStore((s) => s.setMicPermission);

  const start = useCallback(async () => {
    try {
      await recorder.current.start(onChunk);
      setRecording(true);
      setMicPermission("granted");
    } catch {
      setMicPermission("denied");
    }
  }, [onChunk, setRecording, setMicPermission]);

  const stop = useCallback(() => {
    recorder.current.stop();
    setRecording(false);
  }, [setRecording]);

  return { start, stop };
}
