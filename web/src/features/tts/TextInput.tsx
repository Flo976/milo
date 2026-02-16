interface Props {
  value: string;
  onChange: (v: string) => void;
  maxLength: number;
  disabled?: boolean;
}

export default function TextInput({
  value,
  onChange,
  maxLength,
  disabled,
}: Props) {
  return (
    <div className="space-y-1">
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        maxLength={maxLength}
        disabled={disabled}
        placeholder="Soraty eto ny lahateny..."
        rows={4}
        className="w-full resize-none rounded-lg border border-gray-700 bg-gray-800 px-4 py-3 text-sm text-gray-100 placeholder-gray-500 focus:border-milo-500 focus:outline-none focus:ring-1 focus:ring-milo-500 disabled:opacity-50"
      />
      <p className="text-right text-xs text-gray-500">
        {value.length}/{maxLength}
      </p>
    </div>
  );
}
