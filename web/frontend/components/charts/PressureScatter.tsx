"use client";

import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { ScatterPoint } from "@/lib/api";

interface PressureScatterProps {
  points: ScatterPoint[];
  modelA: string;
  modelB: string;
}

export function PressureScatter({ points, modelA, modelB }: PressureScatterProps) {
  // Separate points by model
  const modelAPoints = points
    .filter((p) => p.model === modelA)
    .map((p) => ({
      x: p.time_remaining,
      y: p.move_time,
      tokens: p.thinking_tokens,
    }));

  const modelBPoints = points
    .filter((p) => p.model === modelB)
    .map((p) => ({
      x: p.time_remaining,
      y: p.move_time,
      tokens: p.thinking_tokens,
    }));

  return (
    <div className="w-full">
      <ResponsiveContainer width="100%" height={350}>
        <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2e2e3e" />
          <XAxis
            type="number"
            dataKey="x"
            name="Time Remaining"
            unit="s"
            stroke="#6b7280"
            tick={{ fill: "#9ca3af", fontSize: 12 }}
            domain={[0, "dataMax"]}
            label={{
              value: "Time Remaining (seconds)",
              position: "insideBottom",
              offset: -10,
              fill: "#9ca3af",
            }}
          />
          <YAxis
            type="number"
            dataKey="y"
            name="Move Time"
            unit="s"
            stroke="#6b7280"
            tick={{ fill: "#9ca3af", fontSize: 12 }}
            label={{
              value: "Move Time (s)",
              angle: -90,
              position: "insideLeft",
              fill: "#9ca3af",
            }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#12121a",
              border: "1px solid #2e2e3e",
              borderRadius: "8px",
            }}
            labelStyle={{ color: "#9ca3af" }}
            formatter={(value: number, name: string) => [
              `${value.toFixed(1)}s`,
              name === "x" ? "Time Remaining" : "Move Time",
            ]}
          />
          <Legend />

          {/* Reference lines for pressure thresholds */}
          <ReferenceLine
            x={30}
            stroke="#ef4444"
            strokeDasharray="3 3"
            label={{ value: "Critical", fill: "#ef4444", fontSize: 10 }}
          />
          <ReferenceLine
            x={60}
            stroke="#f59e0b"
            strokeDasharray="3 3"
            label={{ value: "High", fill: "#f59e0b", fontSize: 10 }}
          />
          <ReferenceLine
            x={120}
            stroke="#22c55e"
            strokeDasharray="3 3"
            label={{ value: "Medium", fill: "#22c55e", fontSize: 10 }}
          />

          <Scatter
            name={modelA}
            data={modelAPoints}
            fill="#6366f1"
            opacity={0.7}
          />
          <Scatter
            name={modelB}
            data={modelBPoints}
            fill="#f43f5e"
            opacity={0.7}
          />
        </ScatterChart>
      </ResponsiveContainer>

      <div className="mt-4 text-sm text-gray-400 text-center">
        Each point represents a move. X-axis shows time remaining when the move started.
        Models that adapt should show lower Y values (faster moves) as X decreases.
      </div>
    </div>
  );
}

