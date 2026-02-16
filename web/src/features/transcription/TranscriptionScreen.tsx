import { useTranscription } from "../../hooks/useTranscription";
import DropZone from "./DropZone";
import ExportMenu from "./ExportMenu";
import TranscriptResult from "./TranscriptResult";

export default function TranscriptionScreen() {
  const { result, loading, error, transcribe } = useTranscription();

  return (
    <div className="mx-auto max-w-2xl space-y-6">
      <h1 className="text-lg font-semibold">Transcription</h1>

      <DropZone onFile={transcribe} disabled={loading} />

      {loading && (
        <div className="flex items-center gap-2 text-sm text-gray-400">
          <span className="h-4 w-4 animate-spin rounded-full border-2 border-milo-400 border-t-transparent" />
          Transcription en cours...
        </div>
      )}

      {error && (
        <p className="rounded-lg bg-red-900/30 px-4 py-2 text-sm text-red-400">
          {error}
        </p>
      )}

      {result && (
        <>
          <TranscriptResult
            text={result.text}
            confidence={result.confidence}
            duration_ms={result.duration_ms}
            processing_ms={result.processing_ms}
          />
          <ExportMenu text={result.text} />
        </>
      )}
    </div>
  );
}
