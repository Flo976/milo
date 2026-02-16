import { useState, useCallback } from "react";
import { api } from "../api/endpoints";
import { AudioPlayer } from "../audio/player";

const player = new AudioPlayer();

export function useTts() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);

  const synthesize = useCallback(
    async (text: string, language: string = "mg") => {
      setLoading(true);
      setError(null);
      setAudioBlob(null);

      try {
        const res = await api.tts({ text, language, format: "wav" });
        const blob = await res.blob();
        setAudioBlob(blob);
        await player.playBlob(blob);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setLoading(false);
      }
    },
    []
  );

  const replay = useCallback(async () => {
    if (audioBlob) {
      await player.playBlob(audioBlob);
    }
  }, [audioBlob]);

  return { loading, error, audioBlob, synthesize, replay };
}
