import { useEffect, useRef } from "react";

interface Props {
  blob: Blob;
}

export default function AudioPreview({ blob }: Props) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const urlRef = useRef<string>("");

  useEffect(() => {
    urlRef.current = URL.createObjectURL(blob);
    return () => URL.revokeObjectURL(urlRef.current);
  }, [blob]);

  const download = () => {
    const a = document.createElement("a");
    a.href = urlRef.current;
    a.download = "milo-tts.wav";
    a.click();
  };

  return (
    <div className="card flex items-center gap-4">
      <audio ref={audioRef} src={urlRef.current} controls className="flex-1" />
      <button onClick={download} className="btn-secondary text-xs">
        Telecharger
      </button>
    </div>
  );
}
