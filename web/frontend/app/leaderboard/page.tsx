"use client";

import { useEffect, useState } from "react";
import { api, ModelStats } from "@/lib/api";

export default function LeaderboardPage() {
  const [models, setModels] = useState<ModelStats[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchLeaderboard() {
      try {
        const data = await api.getLeaderboard();
        setModels(data.models);
      } catch (error) {
        console.error("Failed to fetch leaderboard:", error);
      } finally {
        setLoading(false);
      }
    }

    fetchLeaderboard();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="animate-pulse text-gray-400">Loading rankings...</div>
      </div>
    );
  }

  return (
    <div className="space-y-8 animate-fade-in max-w-4xl mx-auto">
      <div className="text-center">
        <h1 className="text-3xl font-bold mb-2">🏆 Model Rankings</h1>
        <p className="text-gray-400">
          ELO ratings calculated from all completed matches
        </p>
      </div>

      {/* Podium for top 3 */}
      {models.length >= 3 && (
        <div className="flex items-end justify-center gap-4 py-8">
          {/* 2nd Place */}
          <div className="text-center animate-slide-up" style={{ animationDelay: "100ms" }}>
            <div className="text-4xl mb-2">🥈</div>
            <div className="card p-4 w-40">
              <div className="font-bold truncate">{models[1].display_name}</div>
              <div className="text-2xl font-mono font-bold text-arena-silver">
                {models[1].elo}
              </div>
              <div className="text-xs text-gray-500">
                {models[1].wins}W-{models[1].losses}L
              </div>
            </div>
            <div className="h-24 bg-arena-silver/20 rounded-t-lg mt-2" />
          </div>

          {/* 1st Place */}
          <div className="text-center animate-slide-up" style={{ animationDelay: "0ms" }}>
            <div className="text-5xl mb-2">🥇</div>
            <div className="card p-4 w-48 border-arena-gold/50">
              <div className="font-bold truncate">{models[0].display_name}</div>
              <div className="text-3xl font-mono font-bold text-arena-gold">
                {models[0].elo}
              </div>
              <div className="text-xs text-gray-500">
                {models[0].wins}W-{models[0].losses}L
              </div>
            </div>
            <div className="h-32 bg-arena-gold/20 rounded-t-lg mt-2" />
          </div>

          {/* 3rd Place */}
          <div className="text-center animate-slide-up" style={{ animationDelay: "200ms" }}>
            <div className="text-4xl mb-2">🥉</div>
            <div className="card p-4 w-40">
              <div className="font-bold truncate">{models[2].display_name}</div>
              <div className="text-2xl font-mono font-bold text-arena-bronze">
                {models[2].elo}
              </div>
              <div className="text-xs text-gray-500">
                {models[2].wins}W-{models[2].losses}L
              </div>
            </div>
            <div className="h-16 bg-arena-bronze/20 rounded-t-lg mt-2" />
          </div>
        </div>
      )}

      {/* Full Leaderboard Table */}
      <div className="card overflow-hidden">
        <table className="w-full">
          <thead className="bg-arena-border/50">
            <tr>
              <th className="px-4 py-3 text-left text-sm font-medium text-gray-400">
                Rank
              </th>
              <th className="px-4 py-3 text-left text-sm font-medium text-gray-400">
                Model
              </th>
              <th className="px-4 py-3 text-right text-sm font-medium text-gray-400">
                ELO
              </th>
              <th className="px-4 py-3 text-right text-sm font-medium text-gray-400">
                Games
              </th>
              <th className="px-4 py-3 text-right text-sm font-medium text-gray-400">
                W-L-D
              </th>
              <th className="px-4 py-3 text-right text-sm font-medium text-gray-400">
                Win Rate
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-arena-border">
            {models.map((model, index) => (
              <tr
                key={model.model_id}
                className="hover:bg-arena-border/30 transition-colors animate-slide-up"
                style={{ animationDelay: `${index * 30}ms` }}
              >
                <td className="px-4 py-4 font-mono text-gray-400">
                  #{index + 1}
                </td>
                <td className="px-4 py-4">
                  <div className="font-medium text-white">
                    {model.display_name}
                  </div>
                </td>
                <td className="px-4 py-4 text-right">
                  <span className="font-mono font-bold text-white">
                    {model.elo}
                  </span>
                  {model.elo_change !== 0 && (
                    <span
                      className={`ml-2 text-sm ${
                        model.elo_change > 0 ? "text-arena-win" : "text-arena-loss"
                      }`}
                    >
                      {model.elo_change > 0 ? "▲" : "▼"}
                      {Math.abs(model.elo_change)}
                    </span>
                  )}
                </td>
                <td className="px-4 py-4 text-right text-gray-400">
                  {model.games_played}
                </td>
                <td className="px-4 py-4 text-right font-mono text-sm">
                  <span className="text-arena-win">{model.wins}</span>-
                  <span className="text-arena-loss">{model.losses}</span>-
                  <span className="text-arena-draw">{model.draws}</span>
                </td>
                <td className="px-4 py-4 text-right">
                  <span className="text-gray-300">
                    {(model.win_rate * 100).toFixed(1)}%
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {models.length === 0 && (
        <div className="text-center py-12 text-gray-400">
          No rankings yet. Complete some matches to build the leaderboard!
        </div>
      )}
    </div>
  );
}

