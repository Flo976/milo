import {
  exportToTxt,
  exportToSrt,
  exportToJson,
  downloadFile,
} from "../../lib/export";

interface Props {
  text: string;
}

export default function ExportMenu({ text }: Props) {
  const msg = {
    id: "1",
    role: "assistant" as const,
    text,
    timestamp: Date.now(),
  };

  return (
    <div className="flex gap-2">
      <button
        className="btn-secondary text-xs"
        onClick={() => downloadFile(exportToTxt([msg]), "transcription.txt", "text/plain")}
      >
        TXT
      </button>
      <button
        className="btn-secondary text-xs"
        onClick={() =>
          downloadFile(exportToSrt([msg]), "transcription.srt", "text/plain")
        }
      >
        SRT
      </button>
      <button
        className="btn-secondary text-xs"
        onClick={() =>
          downloadFile(
            exportToJson([msg]),
            "transcription.json",
            "application/json"
          )
        }
      >
        JSON
      </button>
    </div>
  );
}
