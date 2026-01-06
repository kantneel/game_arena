"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { api, ModelProfile, PressureStats } from "@/lib/api";
import {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";

export default function ModelProfilePage() {
  const params = useParams();
  const modelId = decodeURIComponent(params.modelId as string);
  const [profile, setProfile] = useState<ModelProfile | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchProfile() {
      try {
        const data = await api.getModelProfile(modelId);
        setProfile(data);
      } catch (error) {
        console.error("Failed to fetch model profile:", error);
      } finally {
        setLoading(false);
      }
    }

    if (modelId) {
      fetchProfile();
    }
  }, [modelId]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="animate-pulse text-gray-400">Loading profile...</div>
      </div>
    );
  }

  if (!profile) {
    return (
      <div className="text-center py-16">
        <h1 className="text-2xl font-bold text-gray-300">Model not found</h1>
        <Link href="/models" className="text-arena-accent hover:underline mt-4 block">
          ← Back to models
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <Link
            href="/models"
            className="text-sm text-gray-400 hover:text-white transition-colors mb-2 block"
          >
            ← All Models
          </Link>
          <h1 className="text-3xl font-bold">{profile.display_name}</h1>
          <div className="text-sm text-gray-500 font-mono mt-1">{profile.model_id}</div>
        </div>
        <div className="text-right">
          <div className="text-4xl font-bold">{profile.elo}</div>
          <div className="text-sm text-gray-400">ELO Rating</div>
        </div>
      </div>

      {/* Stats Overview */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <StatCard
          value={profile.total_matches}
          label="Matches"
          color="text-white"
        />
        <StatCard
          value={profile.wins}
          label="Wins"
          color="text-arena-win"
        />
        <StatCard
          value={profile.losses}
          label="Losses"
          color="text-arena-loss"
        />
        <StatCard
          value={profile.draws}
          label="Draws"
          color="text-gray-400"
        />
        <StatCard
          value={`${(profile.win_rate * 100).toFixed(1)}%`}
          label="Win Rate"
          color={profile.win_rate >= 0.5 ? "text-arena-win" : "text-arena-loss"}
        />
      </div>

      {/* Time Pressure Behavior */}
      <div className="grid lg:grid-cols-2 gap-6">
        {/* Adaptation Metrics */}
        <div className="card p-6">
          <h2 className="text-xl font-bold mb-4">Time Pressure Behavior</h2>
          <div className="space-y-4">
            <AdaptationMetric
              label="Speed Adaptation"
              value={profile.speed_adaptation_ratio}
              description="How move time changes under pressure"
              interpretation={
                profile.speed_adaptation_ratio < 0.7
                  ? "Significantly speeds up"
                  : profile.speed_adaptation_ratio < 0.9
                  ? "Moderately adapts"
                  : "Maintains consistent pace"
              }
            />
            <AdaptationMetric
              label="Quality Preservation"
              value={profile.quality_degradation_ratio}
              description="How move quality changes under pressure"
              interpretation={
                profile.quality_degradation_ratio < 1.5
                  ? "Maintains quality"
                  : profile.quality_degradation_ratio < 2.5
                  ? "Some quality drop"
                  : "Significant blunders"
              }
            />
            <AdaptationMetric
              label="Thinking Reduction"
              value={profile.thinking_reduction_ratio}
              description="How thinking depth changes under pressure"
              interpretation={
                profile.thinking_reduction_ratio < 0.5
                  ? "Dramatically reduces thinking"
                  : profile.thinking_reduction_ratio < 0.8
                  ? "Moderately reduces"
                  : "Maintains thinking depth"
              }
            />
          </div>
        </div>

        {/* Pressure Stats Chart */}
        <div className="card p-6">
          <h2 className="text-xl font-bold mb-4">Performance by Pressure Level</h2>
          {profile.pressure_stats.length > 0 ? (
            <PressureStatsChart stats={profile.pressure_stats} />
          ) : (
            <div className="text-center py-8 text-gray-400">
              Not enough data for pressure analysis
            </div>
          )}
        </div>
      </div>

      {/* Move Time Stats */}
      <div className="card p-6">
        <h2 className="text-xl font-bold mb-4">Time Management</h2>
        <div className="grid md:grid-cols-3 gap-6">
          <div className="text-center p-4 bg-arena-border/30 rounded-lg">
            <div className="text-3xl font-mono font-bold">
              {profile.avg_move_time.toFixed(1)}s
            </div>
            <div className="text-sm text-gray-400 mt-1">Average Move Time</div>
          </div>
          <div className="text-center p-4 bg-arena-border/30 rounded-lg">
            <div className="text-3xl font-mono font-bold">
              {profile.avg_thinking_tokens?.toLocaleString() || "—"}
            </div>
            <div className="text-sm text-gray-400 mt-1">Avg Thinking Tokens</div>
          </div>
          <div className="text-center p-4 bg-arena-border/30 rounded-lg">
            <div className="text-3xl font-mono font-bold">{profile.total_moves}</div>
            <div className="text-sm text-gray-400 mt-1">Total Moves Played</div>
          </div>
        </div>
      </div>

      {/* Recent Matches */}
      <div>
        <h2 className="text-xl font-bold mb-4">Recent Matches</h2>
        <div className="card overflow-hidden">
          <table className="w-full">
            <thead className="bg-arena-border/50">
              <tr>
                <th className="px-4 py-3 text-left text-sm font-medium text-gray-400">
                  Date
                </th>
                <th className="px-4 py-3 text-left text-sm font-medium text-gray-400">
                  Opponent
                </th>
                <th className="px-4 py-3 text-center text-sm font-medium text-gray-400">
                  Result
                </th>
                <th className="px-4 py-3 text-center text-sm font-medium text-gray-400">
                  Score
                </th>
                <th className="px-4 py-3"></th>
              </tr>
            </thead>
            <tbody className="divide-y divide-arena-border">
              {profile.recent_matches.map((match) => (
                <tr
                  key={match.match_id}
                  className="hover:bg-arena-border/30 transition-colors"
                >
                  <td className="px-4 py-3 text-sm text-gray-400">
                    {new Date(match.date).toLocaleDateString()}
                  </td>
                  <td className="px-4 py-3">{match.opponent}</td>
                  <td className="px-4 py-3 text-center">
                    <span
                      className={`px-2 py-1 rounded text-xs font-medium ${
                        match.result === "win"
                          ? "bg-arena-win/20 text-arena-win"
                          : match.result === "loss"
                          ? "bg-arena-loss/20 text-arena-loss"
                          : "bg-gray-500/20 text-gray-400"
                      }`}
                    >
                      {match.result.toUpperCase()}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-center font-mono">
                    {match.wins}-{match.losses}
                    {match.draws > 0 && `-${match.draws}`}
                  </td>
                  <td className="px-4 py-3">
                    <Link
                      href={`/matches/${match.match_id}`}
                      className="text-arena-accent hover:text-arena-accent-dim transition-colors text-sm"
                    >
                      View →
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function StatCard({
  value,
  label,
  color,
}: {
  value: string | number;
  label: string;
  color: string;
}) {
  return (
    <div className="stat-card">
      <div className={`stat-value ${color}`}>{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  );
}

function AdaptationMetric({
  label,
  value,
  description,
  interpretation,
}: {
  label: string;
  value: number;
  description: string;
  interpretation: string;
}) {
  // Color based on ratio (1.0 = neutral, <1 = green/good for speed, >1 = red/bad for quality)
  const isGood = value < 1.0;

  return (
    <div className="flex items-center justify-between p-3 bg-arena-border/30 rounded-lg">
      <div>
        <div className="font-medium">{label}</div>
        <div className="text-xs text-gray-500">{description}</div>
        <div className="text-sm text-gray-400 mt-1">{interpretation}</div>
      </div>
      <div
        className={`text-2xl font-mono font-bold ${
          isGood ? "text-arena-win" : "text-orange-400"
        }`}
      >
        {(value * 100).toFixed(0)}%
      </div>
    </div>
  );
}

function PressureStatsChart({ stats }: { stats: PressureStats[] }) {
  const pressureOrder = ["comfortable", "medium", "high", "critical"];
  
  const data = pressureOrder.map((level) => {
    const stat = stats.find((s) => s.pressure_level === level);
    return {
      pressure: level.charAt(0).toUpperCase() + level.slice(1),
      moves: stat?.move_count || 0,
      avgTime: stat?.avg_move_time || 0,
      blunderRate: (stat?.blunder_rate || 0) * 100,
    };
  });

  return (
    <ResponsiveContainer width="100%" height={250}>
      <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#2e2e3e" />
        <XAxis
          dataKey="pressure"
          stroke="#6b7280"
          tick={{ fill: "#9ca3af", fontSize: 12 }}
        />
        <YAxis
          yAxisId="left"
          stroke="#6b7280"
          tick={{ fill: "#9ca3af", fontSize: 12 }}
          label={{
            value: "Avg Time (s)",
            angle: -90,
            position: "insideLeft",
            fill: "#9ca3af",
          }}
        />
        <YAxis
          yAxisId="right"
          orientation="right"
          stroke="#6b7280"
          tick={{ fill: "#9ca3af", fontSize: 12 }}
          label={{
            value: "Blunder %",
            angle: 90,
            position: "insideRight",
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
        <Bar yAxisId="left" dataKey="avgTime" fill="#6366f1" radius={[4, 4, 0, 0]} name="Avg Time (s)" />
        <Bar yAxisId="right" dataKey="blunderRate" fill="#f43f5e" radius={[4, 4, 0, 0]} name="Blunder %" />
      </BarChart>
    </ResponsiveContainer>
  );
}

