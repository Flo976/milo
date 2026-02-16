import { useState, useCallback } from "react";
import { api } from "../api/endpoints";
import { arrayBufferToBase64 } from "../audio/encoder";

interface TranscriptionResult {
  text: string;
  confidence: number;
  duration_ms: number;
  processing_ms: number;
}

export function useTranscription() {
  const [result, setResult] = useState<TranscriptionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const transcribe = useCallback(async (file: File) => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const buffer = await file.arrayBuffer();
      const b64 = arrayBufferToBase64(buffer);

      const res = await api.stt({
        audio: b64,
        format: "wav",
        sample_rate: 16000,
      });

      setResult({
        text: res.text,
        confidence: res.confidence,
        duration_ms: res.duration_ms,
        processing_ms: res.processing_ms,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  return { result, loading, error, transcribe };
}
