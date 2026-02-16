import { useConversationStore } from "../store/conversationStore";
import { useAudioStore } from "../store/audioStore";

export default function StatusBar() {
  const mode = useConversationStore((s) => s.mode);
  const isRecording = useAudioStore((s) => s.isRecording);

  return (
    <footer className="flex h-8 items-center gap-4 border-t border-gray-800 bg-gray-900 px-4 text-xs text-gray-500">
      <span className="flex items-center gap-1.5">
        <span
          className={`h-2 w-2 rounded-full ${
            isRecording ? "bg-red-500 animate-pulse" : "bg-gray-600"
          }`}
        />
        {isRecording ? "Recording" : "Idle"}
      </span>

      {mode && (
        <span className="flex items-center gap-1.5">
          <span
            className={`h-2 w-2 rounded-full ${
              mode === "cloud" ? "bg-blue-400" : "bg-amber-400"
            }`}
          />
          {mode === "cloud" ? "Cloud" : "Local"}
        </span>
      )}
    </footer>
  );
}
