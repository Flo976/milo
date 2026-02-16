import { clsx } from "clsx";
import type { Message } from "../../api/types";
import { formatTimestamp } from "../../lib/format";

interface Props {
  message: Message;
}

export default function ChatBubble({ message }: Props) {
  const isUser = message.role === "user";

  return (
    <div
      className={clsx(
        "flex w-full",
        isUser ? "justify-end" : "justify-start"
      )}
    >
      <div
        className={clsx(
          "max-w-[80%] rounded-2xl px-4 py-2.5",
          isUser
            ? "rounded-br-md bg-milo-600 text-white"
            : "rounded-bl-md bg-gray-800 text-gray-100"
        )}
      >
        <p className="text-sm leading-relaxed">{message.text}</p>
        <p
          className={clsx(
            "mt-1 text-[10px]",
            isUser ? "text-milo-200" : "text-gray-500"
          )}
        >
          {formatTimestamp(message.timestamp)}
        </p>
      </div>
    </div>
  );
}
