import { useCallback, useState } from "react";
import { clsx } from "clsx";

interface Props {
  onFile: (file: File) => void;
  disabled?: boolean;
}

export default function DropZone({ onFile, disabled }: Props) {
  const [dragging, setDragging] = useState(false);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) onFile(file);
    },
    [onFile]
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) onFile(file);
    },
    [onFile]
  );

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      className={clsx(
        "flex flex-col items-center justify-center rounded-xl border-2 border-dashed p-10 transition-colors",
        dragging
          ? "border-milo-400 bg-milo-400/5"
          : "border-gray-700 hover:border-gray-600",
        disabled && "pointer-events-none opacity-50"
      )}
    >
      <svg
        className="mb-3 h-10 w-10 text-gray-500"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
        strokeWidth={1.5}
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5"
        />
      </svg>
      <p className="mb-1 text-sm text-gray-400">
        Glissez un fichier audio ici
      </p>
      <p className="text-xs text-gray-600">WAV, MP3 - max 10 Mo</p>

      <label className="btn-secondary mt-4 cursor-pointer text-sm">
        Choisir un fichier
        <input
          type="file"
          accept="audio/*"
          onChange={handleChange}
          className="hidden"
          disabled={disabled}
        />
      </label>
    </div>
  );
}
