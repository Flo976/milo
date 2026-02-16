import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

interface DataPoint {
  time: string;
  p50: number;
  p95: number;
}

interface Props {
  data: DataPoint[];
}

export default function LatencyChart({ data }: Props) {
  return (
    <div className="card">
      <h3 className="mb-4 text-sm font-medium text-gray-300">
        Latence (ms)
      </h3>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="time" stroke="#6b7280" fontSize={11} />
            <YAxis stroke="#6b7280" fontSize={11} />
            <Tooltip
              contentStyle={{
                backgroundColor: "#1f2937",
                border: "1px solid #374151",
                borderRadius: "8px",
              }}
            />
            <Line
              type="monotone"
              dataKey="p50"
              stroke="#4ade80"
              strokeWidth={2}
              dot={false}
              name="p50"
            />
            <Line
              type="monotone"
              dataKey="p95"
              stroke="#f59e0b"
              strokeWidth={2}
              dot={false}
              name="p95"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
