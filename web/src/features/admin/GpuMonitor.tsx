interface Props {
  vram: {
    available: boolean;
    allocated_gb?: number;
    total_gb?: number;
    free_gb?: number;
  };
}

export default function GpuMonitor({ vram }: Props) {
  if (!vram.available) {
    return (
      <div className="card">
        <h3 className="text-sm font-medium text-gray-300">GPU</h3>
        <p className="mt-2 text-sm text-gray-500">GPU non disponible</p>
      </div>
    );
  }

  const used = vram.allocated_gb ?? 0;
  const total = vram.total_gb ?? 16;
  const pct = total > 0 ? (used / total) * 100 : 0;

  return (
    <div className="card">
      <h3 className="mb-3 text-sm font-medium text-gray-300">GPU VRAM</h3>

      <div className="mb-2 h-3 overflow-hidden rounded-full bg-gray-800">
        <div
          className={`h-full rounded-full transition-all ${
            pct > 87 ? "bg-red-500" : pct > 62 ? "bg-amber-500" : "bg-milo-500"
          }`}
          style={{ width: `${pct}%` }}
        />
      </div>

      <div className="flex justify-between text-xs text-gray-500">
        <span>{used.toFixed(1)} GB utilise</span>
        <span>{total.toFixed(1)} GB total</span>
      </div>
    </div>
  );
}
