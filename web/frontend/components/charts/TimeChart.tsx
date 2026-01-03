"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { MoveRecord } from "@/lib/api";

interface TimeChartProps {
  moves: MoveRecord[];
}

export function TimeChart({ moves }: TimeChartProps) {
  // Prepare data for the chart
  const chartData = moves.map((move, index) => ({
    moveNumber: index + 1,
    player: move.player,
    color: move.color,
    timeTaken: move.time_taken,
    timeRemaining: move.time_remaining,
  }));

  // Separate by color for two-line chart
  const whiteMoves = chartData.filter((d) => d.color === "white");
  const blackMoves = chartData.filter((d) => d.color === "black");

  // Merge into move pairs
  const combinedData = [];
  const maxMoves = Math.max(whiteMoves.length, blackMoves.length);

  for (let i = 0; i < maxMoves; i++) {
    combinedData.push({
      moveNum: i + 1,
      whiteTime: whiteMoves[i]?.timeRemaining ?? null,
      blackTime: blackMoves[i]?.timeRemaining ?? null,
      whiteMoveDuration: whiteMoves[i]?.timeTaken ?? null,
      blackMoveDuration: blackMoves[i]?.timeTaken ?? null,
    });
  }

  return (
    <div className="space-y-6">
      {/* Time Remaining Chart */}
      <div>
        <h3 className="text-sm font-medium text-gray-400 mb-3">
          Time Remaining
        </h3>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={combinedData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2e2e3e" />
            <XAxis
              dataKey="moveNum"
              stroke="#6b7280"
              tick={{ fill: "#9ca3af", fontSize: 12 }}
              label={{
                value: "Move",
                position: "insideBottom",
                offset: -5,
                fill: "#9ca3af",
              }}
            />
            <YAxis
              stroke="#6b7280"
              tick={{ fill: "#9ca3af", fontSize: 12 }}
              label={{
                value: "Seconds",
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
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="whiteTime"
              name="White"
              stroke="#f8fafc"
              strokeWidth={2}
              dot={false}
              connectNulls
            />
            <Line
              type="monotone"
              dataKey="blackTime"
              name="Black"
              stroke="#6366f1"
              strokeWidth={2}
              dot={false}
              connectNulls
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Move Duration Chart */}
      <div>
        <h3 className="text-sm font-medium text-gray-400 mb-3">
          Time per Move
        </h3>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={combinedData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2e2e3e" />
            <XAxis
              dataKey="moveNum"
              stroke="#6b7280"
              tick={{ fill: "#9ca3af", fontSize: 12 }}
            />
            <YAxis
              stroke="#6b7280"
              tick={{ fill: "#9ca3af", fontSize: 12 }}
              label={{
                value: "Seconds",
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
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="whiteMoveDuration"
              name="White (time/move)"
              stroke="#f8fafc"
              strokeWidth={2}
              dot={{ r: 2 }}
              connectNulls
            />
            <Line
              type="monotone"
              dataKey="blackMoveDuration"
              name="Black (time/move)"
              stroke="#6366f1"
              strokeWidth={2}
              dot={{ r: 2 }}
              connectNulls
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

