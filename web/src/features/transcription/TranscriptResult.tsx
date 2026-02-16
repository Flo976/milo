import { formatMs } from "../../lib/format";

interface Props {
  text: string;
  confidence: number;
  duration_ms: number;
  processing_ms: number;
}

export default function TranscriptResult({
  text,
  confidence,
  duration_ms,
  processing_ms,
}: Props) {
  return (
    <div className="card space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-gray-300">Resultat</h3>
        <div className="flex gap-3 text-xs text-gray-500">
          <span>Confiance: {Math.round(confidence * 100)}%</span>
          <span>Duree: {formatMs(duration_ms)}</span>
          <span>Traitement: {formatMs(processing_ms)}</span>
        </div>
      </div>
      <p className="rounded-lg bg-gray-800 p-4 text-sm leading-relaxed text-gray-100">
        {text}
      </p>
    </div>
  );
}
