import { useConversationStore } from "../../store/conversationStore";

export default function ModeIndicator() {
  const mode = useConversationStore((s) => s.mode);

  if (!mode) return null;

  const config = {
    cloud: { color: "bg-blue-400", label: "Cloud boost" },
    local: { color: "bg-amber-400", label: "Local" },
    echo: { color: "bg-gray-400", label: "Echo (no LLM)" },
  }[mode] ?? { color: "bg-gray-400", label: mode };

  return (
    <span className="inline-flex items-center gap-1.5 rounded-full bg-gray-800 px-3 py-1 text-xs font-medium">
      <span className={`h-2 w-2 rounded-full ${config.color}`} />
      {config.label}
    </span>
  );
}
