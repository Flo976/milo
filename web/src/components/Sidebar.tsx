import { useState } from "react";
import { NavLink } from "react-router-dom";
import { clsx } from "clsx";
import { useSettingsStore } from "../store/settingsStore";

const links = [
  { to: "/", label: "Conversation", icon: "M" },
  { to: "/transcription", label: "Transcription", icon: "T" },
  { to: "/tts", label: "Synthese", icon: "S" },
  { to: "/admin", label: "Admin", icon: "A" },
];

export default function Sidebar() {
  const apiKey = useSettingsStore((s) => s.apiKey);
  const setApiKey = useSettingsStore((s) => s.setApiKey);
  const [showSettings, setShowSettings] = useState(!apiKey);
  const [keyInput, setKeyInput] = useState(apiKey);

  const saveKey = () => {
    setApiKey(keyInput.trim());
    if (keyInput.trim()) setShowSettings(false);
  };

  return (
    <aside className="hidden w-56 flex-col border-r border-gray-800 bg-gray-900 md:flex">
      <div className="flex h-14 items-center gap-2 border-b border-gray-800 px-4">
        <span className="text-xl font-bold text-milo-400">Milo</span>
        <span className="text-xs text-gray-500">Voice</span>
      </div>

      <nav className="flex-1 space-y-1 p-3">
        {links.map((l) => (
          <NavLink
            key={l.to}
            to={l.to}
            end={l.to === "/"}
            className={({ isActive }) =>
              clsx(
                "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                isActive
                  ? "bg-milo-600/20 text-milo-400"
                  : "text-gray-400 hover:bg-gray-800 hover:text-gray-200"
              )
            }
          >
            <span className="flex h-7 w-7 items-center justify-center rounded-md bg-gray-800 text-xs font-bold">
              {l.icon}
            </span>
            {l.label}
          </NavLink>
        ))}
      </nav>

      {/* Settings panel */}
      <div className="border-t border-gray-800 p-3">
        {showSettings ? (
          <div className="space-y-2">
            <label className="text-xs text-gray-500">API Key</label>
            <input
              type="password"
              value={keyInput}
              onChange={(e) => setKeyInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && saveKey()}
              placeholder="Bearer token..."
              className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-xs text-gray-200 focus:border-milo-500 focus:outline-none"
            />
            <button onClick={saveKey} className="btn-primary w-full text-xs py-1.5">
              Connecter
            </button>
          </div>
        ) : (
          <button
            onClick={() => setShowSettings(true)}
            className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-xs text-gray-400 hover:bg-gray-800 hover:text-gray-200"
          >
            <span className="h-2 w-2 rounded-full bg-milo-500" />
            Connecte
          </button>
        )}
      </div>

      <div className="border-t border-gray-800 px-4 py-2 text-xs text-gray-600">
        v0.1.0
      </div>
    </aside>
  );
}
