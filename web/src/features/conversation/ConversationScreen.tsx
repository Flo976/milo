import { useConversation } from "../../hooks/useConversation";
import { useConversationStore } from "../../store/conversationStore";
import ChatHistory from "./ChatHistory";
import LiveTranscript from "./LiveTranscript";
import MicButton from "./MicButton";
import ModeIndicator from "./ModeIndicator";
import SoundWave from "./SoundWave";

export default function ConversationScreen() {
  const { isRecording, toggleRecording } = useConversation();
  const isSpeaking = useConversationStore((s) => s.isSpeaking);

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-gray-800 pb-3">
        <h1 className="text-lg font-semibold">Conversation</h1>
        <ModeIndicator />
      </div>

      <ChatHistory />

      <div className="space-y-3 border-t border-gray-800 pt-4">
        <LiveTranscript />

        <div className="flex flex-col items-center gap-3">
          <SoundWave active={isRecording && isSpeaking} />
          <MicButton
            isRecording={isRecording}
            onClick={() => void toggleRecording()}
          />
          <p className="text-xs text-gray-500">
            {isRecording ? "Parlez en malagasy..." : "Appuyez pour parler"}
          </p>
        </div>
      </div>
    </div>
  );
}
