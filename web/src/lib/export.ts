import type { Message } from "../api/types";

export function exportToTxt(messages: Message[]): string {
  return messages
    .map(
      (m) =>
        `[${new Date(m.timestamp).toISOString()}] ${m.role === "user" ? "Utilisateur" : "Milo"}: ${m.text}`
    )
    .join("\n");
}

export function exportToSrt(messages: Message[]): string {
  return messages
    .map((m, i) => {
      const start = formatSrtTime(i * 3000);
      const end = formatSrtTime((i + 1) * 3000);
      return `${i + 1}\n${start} --> ${end}\n${m.text}\n`;
    })
    .join("\n");
}

function formatSrtTime(ms: number): string {
  const h = Math.floor(ms / 3600000);
  const m = Math.floor((ms % 3600000) / 60000);
  const s = Math.floor((ms % 60000) / 1000);
  const f = ms % 1000;
  return `${pad(h)}:${pad(m)}:${pad(s)},${pad3(f)}`;
}

function pad(n: number): string {
  return String(n).padStart(2, "0");
}

function pad3(n: number): string {
  return String(n).padStart(3, "0");
}

export function exportToJson(messages: Message[]): string {
  return JSON.stringify(
    messages.map((m) => ({
      role: m.role,
      text: m.text,
      timestamp: new Date(m.timestamp).toISOString(),
    })),
    null,
    2
  );
}

export function downloadFile(content: string, filename: string, mime: string) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
