import { useHealth } from "../../hooks/useHealth";
import GpuMonitor from "./GpuMonitor";
import LatencyChart from "./LatencyChart";

// Placeholder data — will be replaced by Prometheus queries
const SAMPLE_LATENCY = [
  { time: "10:00", p50: 350, p95: 800 },
  { time: "10:05", p50: 320, p95: 750 },
  { time: "10:10", p50: 380, p95: 900 },
  { time: "10:15", p50: 340, p95: 820 },
  { time: "10:20", p50: 310, p95: 700 },
  { time: "10:25", p50: 360, p95: 850 },
];

export default function AdminDashboard() {
  const { health, error } = useHealth(10000);

  return (
    <div className="space-y-6">
      <h1 className="text-lg font-semibold">Dashboard Admin</h1>

      {error && (
        <p className="rounded-lg bg-red-900/30 px-4 py-2 text-sm text-red-400">
          API indisponible: {error}
        </p>
      )}

      {/* Status cards */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <div className="card">
          <p className="text-xs text-gray-500">Status</p>
          <p className="mt-1 text-xl font-bold">
            {health ? (
              <span
                className={
                  health.status === "healthy"
                    ? "text-milo-400"
                    : "text-amber-400"
                }
              >
                {health.status}
              </span>
            ) : (
              <span className="text-gray-600">...</span>
            )}
          </p>
        </div>

        <div className="card">
          <p className="text-xs text-gray-500">Mode</p>
          <p className="mt-1 text-xl font-bold capitalize">
            {health?.mode ?? "..."}
          </p>
        </div>

        <div className="card">
          <p className="text-xs text-gray-500">Modeles charges</p>
          <p className="mt-1 text-xl font-bold">
            {health?.models_loaded.length ?? 0}
          </p>
          {health && (
            <p className="mt-1 text-xs text-gray-500">
              {health.models_loaded.join(", ") || "Aucun"}
            </p>
          )}
        </div>

        <div className="card">
          <p className="text-xs text-gray-500">Redis</p>
          <p className="mt-1 text-xl font-bold">
            {health?.redis_connected ? (
              <span className="text-milo-400">Connecte</span>
            ) : (
              <span className="text-red-400">Deconnecte</span>
            )}
          </p>
        </div>
      </div>

      {/* GPU */}
      {health && <GpuMonitor vram={health.vram} />}

      {/* Latency chart */}
      <LatencyChart data={SAMPLE_LATENCY} />
    </div>
  );
}
