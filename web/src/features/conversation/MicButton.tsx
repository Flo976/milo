import { clsx } from "clsx";

interface Props {
  isRecording: boolean;
  onClick: () => void;
  disabled?: boolean;
}

export default function MicButton({ isRecording, onClick, disabled }: Props) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      aria-label={isRecording ? "Stop recording" : "Start recording"}
      className={clsx(
        "relative flex h-20 w-20 items-center justify-center rounded-full transition-all",
        "focus:outline-none focus:ring-2 focus:ring-milo-400 focus:ring-offset-2 focus:ring-offset-gray-950",
        isRecording
          ? "bg-red-600 shadow-lg shadow-red-600/30 hover:bg-red-500"
          : "bg-milo-600 shadow-lg shadow-milo-600/30 hover:bg-milo-500",
        disabled && "cursor-not-allowed opacity-50"
      )}
    >
      {isRecording ? (
        <svg className="h-8 w-8 text-white" viewBox="0 0 24 24" fill="currentColor">
          <rect x="6" y="6" width="12" height="12" rx="2" />
        </svg>
      ) : (
        <svg className="h-8 w-8 text-white" viewBox="0 0 24 24" fill="currentColor">
          <path d="M12 14a3 3 0 0 0 3-3V5a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3z" />
          <path d="M17 11a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.93V21h2v-3.07A7 7 0 0 0 19 11h-2z" />
        </svg>
      )}

      {isRecording && (
        <span className="absolute inset-0 animate-ping rounded-full bg-red-600 opacity-30" />
      )}
    </button>
  );
}
