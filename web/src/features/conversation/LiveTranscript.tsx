import { useConversationStore } from "../../store/conversationStore";

export default function LiveTranscript() {
  const transcript = useConversationStore((s) => s.liveTranscript);

  if (!transcript) return null;

  return (
    <div className="rounded-lg border border-gray-700 bg-gray-800/50 px-4 py-2">
      <p className="text-sm italic text-gray-400">
        <span className="mr-1 inline-block h-2 w-2 animate-pulse rounded-full bg-milo-400" />
        {transcript}
      </p>
    </div>
  );
}
