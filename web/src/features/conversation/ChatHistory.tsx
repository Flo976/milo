import { useEffect, useRef } from "react";
import { useConversationStore } from "../../store/conversationStore";
import ChatBubble from "./ChatBubble";

export default function ChatHistory() {
  const messages = useConversationStore((s) => s.messages);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center text-gray-600">
        <p className="text-lg font-medium">Manao ahoana!</p>
        <p className="mt-1 text-sm">
          Appuyez sur le micro pour parler en malagasy
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-1 flex-col gap-3 overflow-y-auto px-2 py-4">
      {messages.map((m) => (
        <ChatBubble key={m.id} message={m} />
      ))}
      <div ref={endRef} />
    </div>
  );
}
