"use client";

import { ModelStats } from "@/lib/api";

interface LeaderboardPreviewProps {
  models: ModelStats[];
}

export function LeaderboardPreview({ models }: LeaderboardPreviewProps) {
  return (
    <div className="card divide-y divide-arena-border">
      {models.map((model, index) => (
        <div
          key={model.model_id}
          className="flex items-center gap-4 p-4 hover:bg-arena-border/30 transition-colors"
        >
          {/* Rank */}
          <div className="w-8 text-center">
            {index === 0 && <span className="text-xl">🥇</span>}
            {index === 1 && <span className="text-xl">🥈</span>}
            {index === 2 && <span className="text-xl">🥉</span>}
            {index > 2 && (
              <span className="text-gray-500 font-mono">{index + 1}</span>
            )}
          </div>

          {/* Model Info */}
          <div className="flex-1 min-w-0">
            <div className="font-medium text-white truncate">
              {model.display_name}
            </div>
            <div className="text-xs text-gray-500">
              {model.wins}W - {model.losses}L - {model.draws}D
            </div>
          </div>

          {/* ELO */}
          <div className="text-right">
            <div className="font-bold font-mono text-white">{model.elo}</div>
            {model.elo_change !== 0 && (
              <div
                className={`text-xs font-mono ${
                  model.elo_change > 0 ? "text-arena-win" : "text-arena-loss"
                }`}
              >
                {model.elo_change > 0 ? "+" : ""}
                {model.elo_change}
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

