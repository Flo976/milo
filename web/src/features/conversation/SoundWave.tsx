import { clsx } from "clsx";

interface Props {
  active: boolean;
}

export default function SoundWave({ active }: Props) {
  const bars = [1, 2, 3, 4, 5];

  return (
    <div className="flex h-8 items-end gap-1" aria-hidden="true">
      {bars.map((i) => (
        <div
          key={i}
          className={clsx(
            "w-1 rounded-full bg-milo-400 transition-all",
            active ? "animate-sound-wave" : "h-1"
          )}
          style={{
            animationDelay: active ? `${i * 0.15}s` : undefined,
            height: active ? undefined : "4px",
          }}
        />
      ))}
    </div>
  );
}
