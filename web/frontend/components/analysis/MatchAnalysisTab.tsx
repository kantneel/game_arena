"use client";

import { useEffect, useState } from "react";
import { api, MatchAnalysis, ScatterData, ThinkingByPressure } from "@/lib/api";
import { PressureScatter } from "@/components/charts/PressureScatter";
import { ThinkingTokensBar } from "@/components/charts/ThinkingTokensBar";
import { AdaptationMetrics } from "@/components/charts/AdaptationMetrics";

interface MatchAnalysisTabProps {
  matchId: string;
}

export function MatchAnalysisTab({ matchId }: MatchAnalysisTabProps) {
  const [analysis, setAnalysis] = useState<MatchAnalysis | null>(null);
  const [scatter, setScatter] = useState<ScatterData | null>(null);
  const [thinking, setThinking] = useState<ThinkingByPressure | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchAnalysis() {
      try {
        const [analysisData, scatterData, thinkingData] = await Promise.all([
          api.getMatchAnalysis(matchId),
          api.getPressureScatter(matchId),
          api.getThinkingByPressure(matchId),
        ]);
        setAnalysis(analysisData);
        setScatter(scatterData);
        setThinking(thinkingData);
      } catch (e) {
        setError("Failed to load analysis data. Make sure the match has move data.");
        console.error(e);
      } finally {
        setLoading(false);
      }
    }

    fetchAnalysis();
  }, [matchId]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-pulse text-gray-400">Loading analysis...</div>
      </div>
    );
  }

  if (error || !analysis) {
    return (
      <div className="text-center py-12 text-gray-400">
        <div className="text-4xl mb-4">📊</div>
        <div>{error || "No analysis data available"}</div>
      </div>
    );
  }

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Key Insights */}
      <section>
        <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
          <span>💡</span> Key Insights
        </h3>
        <div className="card p-4">
          <ul className="space-y-2">
            {analysis.insights.map((insight, i) => (
              <li key={i} className="flex items-start gap-2">
                <span className="text-arena-accent mt-1">•</span>
                <span className="text-gray-300">{insight}</span>
              </li>
            ))}
          </ul>
        </div>
      </section>

      {/* Adaptation Metrics */}
      <section>
        <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
          <span>📈</span> Adaptation Metrics
        </h3>
        <AdaptationMetrics modelA={analysis.model_a} modelB={analysis.model_b} />
      </section>

      {/* Time Pressure Scatter */}
      {scatter && scatter.points.length > 0 && (
        <section>
          <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
            <span>⏱️</span> Time Remaining vs Move Time
          </h3>
          <div className="card p-6">
            <PressureScatter
              points={scatter.points}
              modelA={scatter.model_a}
              modelB={scatter.model_b}
            />
          </div>
        </section>
      )}

      {/* Thinking Tokens by Pressure */}
      {thinking && (
        <section>
          <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
            <span>🧠</span> Thinking Depth by Pressure Level
          </h3>
          <div className="card p-6">
            <ThinkingTokensBar data={thinking} />
          </div>
        </section>
      )}

      {/* Pressure Stats Table */}
      <section>
        <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
          <span>📊</span> Detailed Pressure Statistics
        </h3>
        <div className="grid md:grid-cols-2 gap-6">
          <PressureStatsTable
            modelName={analysis.model_a.name}
            stats={analysis.model_a.pressure_stats}
          />
          <PressureStatsTable
            modelName={analysis.model_b.name}
            stats={analysis.model_b.pressure_stats}
          />
        </div>
      </section>
    </div>
  );
}

function PressureStatsTable({
  modelName,
  stats,
}: {
  modelName: string;
  stats: { pressure_level: string; move_count: number; avg_move_time: number; avg_thinking_tokens: number | null; blunder_rate: number }[];
}) {
  const pressureOrder = ["comfortable", "medium", "high", "critical"];
  const orderedStats = pressureOrder
    .map((level) => stats.find((s) => s.pressure_level === level))
    .filter(Boolean);

  return (
    <div className="card overflow-hidden">
      <div className="bg-arena-border/50 px-4 py-2 font-medium">{modelName}</div>
      <table className="w-full text-sm">
        <thead className="bg-arena-border/30">
          <tr>
            <th className="px-3 py-2 text-left text-gray-400">Pressure</th>
            <th className="px-3 py-2 text-right text-gray-400">Moves</th>
            <th className="px-3 py-2 text-right text-gray-400">Avg Time</th>
            <th className="px-3 py-2 text-right text-gray-400">Tokens</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-arena-border">
          {orderedStats.map((stat) => (
            <tr key={stat!.pressure_level} className="hover:bg-arena-border/20">
              <td className="px-3 py-2 capitalize">
                <span
                  className={`inline-block w-2 h-2 rounded-full mr-2 ${
                    stat!.pressure_level === "critical"
                      ? "bg-red-500"
                      : stat!.pressure_level === "high"
                      ? "bg-orange-500"
                      : stat!.pressure_level === "medium"
                      ? "bg-yellow-500"
                      : "bg-green-500"
                  }`}
                />
                {stat!.pressure_level}
              </td>
              <td className="px-3 py-2 text-right font-mono">{stat!.move_count}</td>
              <td className="px-3 py-2 text-right font-mono">
                {stat!.avg_move_time.toFixed(1)}s
              </td>
              <td className="px-3 py-2 text-right font-mono">
                {stat!.avg_thinking_tokens != null
                  ? Math.round(stat!.avg_thinking_tokens).toLocaleString()
                  : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

