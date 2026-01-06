"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { ThinkingByPressure } from "@/lib/api";

interface ThinkingTokensBarProps {
  data: ThinkingByPressure;
}

export function ThinkingTokensBar({ data }: ThinkingTokensBarProps) {
  // Order pressure levels
  const pressureOrder = ["comfortable", "medium", "high", "critical"];
  
  const chartData = pressureOrder.map((pressure) => {
    const item = data.data.find((d) => d.pressure === pressure);
    return {
      pressure: pressure.charAt(0).toUpperCase() + pressure.slice(1),
      [data.model_a]: item?.model_a_avg_tokens || 0,
      [data.model_b]: item?.model_b_avg_tokens || 0,
      model_a_time: item?.model_a_avg_time || 0,
      model_b_time: item?.model_b_avg_time || 0,
    };
  });

  return (
    <div className="w-full">
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2e2e3e" />
          <XAxis
            dataKey="pressure"
            stroke="#6b7280"
            tick={{ fill: "#9ca3af", fontSize: 12 }}
          />
          <YAxis
            stroke="#6b7280"
            tick={{ fill: "#9ca3af", fontSize: 12 }}
            label={{
              value: "Avg Thinking Tokens",
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
            formatter={(value: number) => [value.toLocaleString(), "tokens"]}
          />
          <Legend />
          <Bar dataKey={data.model_a} fill="#6366f1" radius={[4, 4, 0, 0]} />
          <Bar dataKey={data.model_b} fill="#f43f5e" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>

      <div className="mt-4 text-sm text-gray-400 text-center">
        Shows how much thinking each model does at different time pressure levels.
        Decreasing bars suggest the model reduces thinking depth under pressure.
      </div>
    </div>
  );
}

