import { clsx } from "clsx";

interface Props {
  value: string;
  onChange: (lang: string) => void;
}

const languages = [
  { code: "mg", label: "Malagasy" },
  { code: "fr", label: "Francais" },
];

export default function LanguageSelector({ value, onChange }: Props) {
  return (
    <div className="flex gap-2">
      {languages.map((lang) => (
        <button
          key={lang.code}
          onClick={() => onChange(lang.code)}
          className={clsx(
            "rounded-lg px-4 py-2 text-sm font-medium transition-colors",
            value === lang.code
              ? "bg-milo-600 text-white"
              : "bg-gray-800 text-gray-400 hover:bg-gray-700"
          )}
        >
          {lang.label}
        </button>
      ))}
    </div>
  );
}
