"use client";

import { ModelPressureProfile } from "@/lib/api";

interface AdaptationMetricsProps {
  modelA: ModelPressureProfile;
  modelB: ModelPressureProfile;
}

export function AdaptationMetrics({ modelA, modelB }: AdaptationMetricsProps) {
  const metrics = [
    {
      label: "Speed Adaptation",
      description: "How much faster under pressure (< 1.0 = faster)",
      modelAValue: modelA.speed_adaptation_ratio,
      modelBValue: modelB.speed_adaptation_ratio,
      format: (v: number) => `${(v * 100).toFixed(0)}%`,
      goodDirection: "lower" as const,
    },
    {
      label: "Quality Preservation",
      description: "Move quality change under pressure (< 1.0 = better)",
      modelAValue: modelA.quality_degradation_ratio,
      modelBValue: modelB.quality_degradation_ratio,
      format: (v: number) => `${(v * 100).toFixed(0)}%`,
      goodDirection: "lower" as const,
    },
    {
      label: "Thinking Reduction",
      description: "Thinking depth change under pressure (< 1.0 = less)",
      modelAValue: modelA.thinking_reduction_ratio,
      modelBValue: modelB.thinking_reduction_ratio,
      format: (v: number) => `${(v * 100).toFixed(0)}%`,
      goodDirection: "neutral" as const,
    },
  ];

  return (
    <div className="space-y-4">
      {metrics.map((metric) => {
        const aIsBetter =
          metric.goodDirection === "lower"
            ? metric.modelAValue < metric.modelBValue
            : metric.goodDirection === "higher"
            ? metric.modelAValue > metric.modelBValue
            : false;
        const bIsBetter =
          metric.goodDirection === "lower"
            ? metric.modelBValue < metric.modelAValue
            : metric.goodDirection === "higher"
            ? metric.modelBValue > metric.modelAValue
            : false;

        return (
          <div key={metric.label} className="card p-4">
            <div className="flex justify-between items-start mb-2">
              <div>
                <div className="font-medium text-white">{metric.label}</div>
                <div className="text-xs text-gray-500">{metric.description}</div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4 mt-3">
              <div
                className={`p-3 rounded-lg ${
                  aIsBetter ? "bg-arena-win/10 border border-arena-win/30" : "bg-arena-border/50"
                }`}
              >
                <div className="text-xs text-gray-400 truncate">{modelA.name}</div>
                <div
                  className={`text-xl font-mono font-bold ${
                    aIsBetter ? "text-arena-win" : "text-white"
                  }`}
                >
                  {metric.format(metric.modelAValue)}
                </div>
              </div>

              <div
                className={`p-3 rounded-lg ${
                  bIsBetter ? "bg-arena-win/10 border border-arena-win/30" : "bg-arena-border/50"
                }`}
              >
                <div className="text-xs text-gray-400 truncate">{modelB.name}</div>
                <div
                  className={`text-xl font-mono font-bold ${
                    bIsBetter ? "text-arena-win" : "text-white"
                  }`}
                >
                  {metric.format(metric.modelBValue)}
                </div>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

