import { useState } from "react";
import { useTts } from "../../hooks/useTts";
import AudioPreview from "./AudioPreview";
import LanguageSelector from "./LanguageSelector";
import TextInput from "./TextInput";

export default function TtsScreen() {
  const [text, setText] = useState("");
  const [language, setLanguage] = useState("mg");
  const { loading, error, audioBlob, synthesize } = useTts();

  const handleGenerate = () => {
    if (text.trim()) {
      void synthesize(text, language);
    }
  };

  return (
    <div className="mx-auto max-w-2xl space-y-6">
      <h1 className="text-lg font-semibold">Synthese vocale</h1>

      <LanguageSelector value={language} onChange={setLanguage} />
      <TextInput value={text} onChange={setText} maxLength={500} disabled={loading} />

      <button
        onClick={handleGenerate}
        disabled={loading || !text.trim()}
        className="btn-primary w-full"
      >
        {loading ? (
          <span className="flex items-center justify-center gap-2">
            <span className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
            Generation...
          </span>
        ) : (
          "Generer l'audio"
        )}
      </button>

      {error && (
        <p className="rounded-lg bg-red-900/30 px-4 py-2 text-sm text-red-400">
          {error}
        </p>
      )}

      {audioBlob && <AudioPreview blob={audioBlob} />}
    </div>
  );
}
